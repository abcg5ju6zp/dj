"""
Node materialization related APIs.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from pydantic import Discriminator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, noload, selectinload

from datajunction_server.api.helpers import (
    get_materialization_info,
    get_save_history,
    resolve_column,
)
from datajunction_server.database import Node, NodeRevision
from datajunction_server.database.backfill import Backfill
from datajunction_server.database.column import Column, ColumnAttribute
from datajunction_server.database.history import History
from datajunction_server.database.materialization import Materialization
from datajunction_server.database.user import User
from datajunction_server.errors import (
    DJBackfillConflictException,
    DJBackfillScheduleUncertainException,
    DJDoesNotExistException,
    DJInvalidInputException,
)
from datajunction_server.internal import backfills as backfill_ops
from datajunction_server.internal.access.authentication.http import SecureAPIRouter
from datajunction_server.internal.access.authorization import (
    AccessChecker,
    AccessDenialMode,
    get_access_checker,
)
from datajunction_server.internal.history import ActivityType, EntityType
from datajunction_server.internal.materializations import (
    create_new_materialization,
    schedule_materialization_jobs,
)
from datajunction_server.materialization.jobs import MaterializationJob
from datajunction_server.models.access import ResourceAction
from datajunction_server.models.base import labelize
from datajunction_server.models.cube_materialization import UpsertCubeMaterialization
from datajunction_server.models.materialization import (
    MaterializationConfigInfoUnified,
    MaterializationInfo,
    MaterializationJobTypeEnum,
    MaterializationStrategy,
    UpsertMaterialization,
)
from datajunction_server.models.node import AvailabilityStateInfo
from datajunction_server.models.node_type import NodeType
from datajunction_server.models.partition import (
    BackfillCompletionInput,
    BackfillStatus,
    BackfillStatusInfo,
    PartitionBackfill,
    WorkflowRunState,
    WorkflowStatusInfo,
)
from datajunction_server.naming import amenable_name
from datajunction_server.service_clients import QueryServiceClient
from datajunction_server.typing import UTCDatetime
from datajunction_server.utils import (
    get_current_user,
    get_query_service_client,
    get_session,
    get_settings,
)

_logger = logging.getLogger(__name__)
settings = get_settings()
router = SecureAPIRouter(tags=["materializations"])


@router.get(
    "/materialization/info",
    status_code=200,
    name="Materialization Jobs Info",
)
def materialization_jobs_info() -> JSONResponse:
    """
    Materialization job types and strategies
    """
    return JSONResponse(
        status_code=200,
        content={
            "job_types": [
                value.value.model_dump() for value in MaterializationJobTypeEnum
            ],
            "strategies": [
                {"name": value, "label": labelize(value)}
                for value in MaterializationStrategy
            ],
        },
    )


@router.post(
    "/nodes/{node_name}/materialization/",
    status_code=201,
    name="Insert or Update a Materialization for a Node",
)
async def upsert_materialization(
    node_name: str,
    materialization: Annotated[
        UpsertCubeMaterialization | UpsertMaterialization,
        Discriminator("job"),
    ],
    *,
    session: AsyncSession = Depends(get_session),
    request: Request,
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
    current_user: User = Depends(get_current_user),
    save_history: Callable = Depends(get_save_history),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> JSONResponse:
    """
    Add or update a materialization of the specified node. If a node_name is specified
    for the materialization config, it will always update that named config.
    """
    # Materializing a node is a write on that node, same as deactivate/backfill below.
    # The cube path also runs READ checks while building its config, but those cover
    # the cube's metrics and dimensions, not the node being materialized.
    access_checker.add_request_by_node_name(node_name, ResourceAction.WRITE)
    await access_checker.check(on_denied=AccessDenialMode.RAISE)

    request_headers = dict(request.headers)
    node = await Node.get_by_name(session, node_name, raise_if_not_exists=True)
    if node.type == NodeType.SOURCE:  # type: ignore
        raise DJInvalidInputException(
            http_status_code=HTTPStatus.BAD_REQUEST,
            message=f"Cannot set materialization config for source node `{node_name}`!",
        )
    if node.type == NodeType.CUBE:  # type: ignore
        node = await Node.get_cube_by_name(session, node_name)
    _logger.info(
        "Upserting materialization for node=%s version=%s",
        node.name,  # type: ignore
        node.current_version,  # type: ignore
    )

    current_revision = node.current  # type: ignore
    old_materializations = {mat.name: mat for mat in current_revision.materializations}

    if materialization.strategy == MaterializationStrategy.INCREMENTAL_TIME:  # type: ignore
        if not node.current.temporal_partition_columns():  # type: ignore
            raise DJInvalidInputException(
                http_status_code=HTTPStatus.BAD_REQUEST,
                message="Cannot create materialization with strategy "
                f"`{materialization.strategy}` without specifying a time partition column!",  # type: ignore
            )

    # Create a new materialization
    new_materialization = await create_new_materialization(
        session,
        current_revision,
        materialization,
        access_checker,  # type: ignore
        current_user=current_user,
    )

    # Check to see if a materialization for this engine already exists with the exact same config
    existing_materialization = old_materializations.get(new_materialization.name)
    deactivated_before = False
    if (
        existing_materialization
        and existing_materialization.config == new_materialization.config
    ):
        _logger.info(
            "Existing materialization found for node=%s version=%s",
            node.name,  # type: ignore
            node.current_version,  # type: ignore
        )
        new_materialization.node_revision = None  # type: ignore
        # if the materialization was deactivated before, restore it
        if existing_materialization.deactivated_at is not None:
            deactivated_before = True
            existing_materialization.deactivated_at = None  # type: ignore
            await save_history(
                event=History(
                    entity_type=EntityType.MATERIALIZATION,
                    entity_name=existing_materialization.name,
                    node=node.name,  # type: ignore
                    activity_type=ActivityType.RESTORE,
                    details={},
                    user=current_user.username,
                ),
                session=session,
            )
            await session.commit()
            await session.refresh(existing_materialization)
        existing_materialization_info = query_service_client.get_materialization_info(
            node_name,
            current_revision.version,  # type: ignore
            current_revision.type,
            new_materialization.name,  # type: ignore
            request_headers=request_headers,
        )
        _logger.info(
            "Refresh materialization workflows for node=%s version=%s",
            node.name,  # type: ignore
            node.current_version,  # type: ignore
        )
        await schedule_materialization_jobs(
            session,
            node_revision_id=current_revision.id,
            materialization_names=[new_materialization.name],
            query_service_client=query_service_client,
            request_headers=request_headers,
        )
        return JSONResponse(
            status_code=HTTPStatus.CREATED,
            content={
                "message": (
                    f"The same materialization config with name `{new_materialization.name}` "
                    f"already exists for node `{node_name}` so no update was performed."
                    if not deactivated_before
                    else f"The same materialization config with name `{new_materialization.name}` "
                    f"already exists for node `{node_name}` but was deactivated. It has now been "
                    f"restored."
                ),
                "info": existing_materialization_info.model_dump(),
            },
        )
    # If changes are detected, update the existing or save the new materialization
    if existing_materialization:
        _logger.info(
            "Updating existing materialization for node=%s version=%s",
            node.name,  # type: ignore
            node.current_version,  # type: ignore
        )
        existing_materialization.config = new_materialization.config
        existing_materialization.schedule = new_materialization.schedule
        # The definition changed in place: advance the generation so backfills
        # submitted against the old definition can no longer confirm partitions
        # or publish availability onto this one.
        existing_materialization.generation = (
            existing_materialization.generation or 1
        ) + 1
        new_materialization.node_revision = None  # type: ignore
        new_materialization = existing_materialization
        new_materialization.deactivated_at = None
    else:
        _logger.info(
            "Adding new materialization for node=%s version=%s",
            node.name,  # type: ignore
            node.current_version,  # type: ignore
        )
        unchanged_existing_materializations = [
            config
            for config in current_revision.materializations
            if config.name != new_materialization.name
        ]
        current_revision.materializations = unchanged_existing_materializations + [  # type: ignore
            new_materialization,
        ]

    # This will add the materialization config, the new node rev, and update the node's version.
    session.add(current_revision)
    session.add(node)

    await save_history(
        event=History(
            entity_type=EntityType.MATERIALIZATION,
            node=node.name,  # type: ignore
            entity_name=new_materialization.name,
            activity_type=(
                ActivityType.CREATE
                if not existing_materialization
                else ActivityType.UPDATE
            ),
            details={
                "node": node.name,  # type: ignore
                "materialization": new_materialization.name,
            },
            user=current_user.username,
        ),
        session=session,
    )
    await session.commit()
    _logger.info(
        "Scheduling materialization workflows for node=%s version=%s",
        node.name,  # type: ignore
        node.current_version,  # type: ignore
    )
    materialization_response = await schedule_materialization_jobs(
        session,
        node_revision_id=current_revision.id,
        materialization_names=[new_materialization.name],
        query_service_client=query_service_client,
        request_headers=request_headers,
    )
    return JSONResponse(
        status_code=200,
        content={
            "message": (
                f"Successfully updated materialization config named `{new_materialization.name}` "
                f"for node `{node_name}`"
            ),
            "urls": [output.urls for output in materialization_response.values()],
        },
    )


@router.get(
    "/nodes/{node_name}/materializations/",
    response_model=list[MaterializationConfigInfoUnified],
    name="List Materializations for a Node",
)
async def list_node_materializations(
    node_name: str,
    show_inactive: bool = False,
    include_all_revisions: bool = False,
    *,
    session: AsyncSession = Depends(get_session),
    request: Request,
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
) -> list[MaterializationConfigInfoUnified]:
    """
    Show all materializations configured for the node, with any associated metadata
    like urls from the materialization service, if available.

    show_inactive: bool - Show materializations that have a deactivated_at timestamp set
    include_all_revisions: bool - Show  materializations for all revisions of the node
    """
    request_headers = dict(request.headers)

    # If materializations from all revisions are requested,
    # this includes the joined load to pull all node revisions
    node = await Node.get_by_name(
        session,
        node_name,
        options=[
            joinedload(Node.revisions).options(
                selectinload(NodeRevision.materializations).selectinload(
                    Materialization.backfills,
                ),
                noload(NodeRevision.created_by),
            ),
        ]
        if include_all_revisions
        else [],
        raise_if_not_exists=True,
    )

    return get_materialization_info(
        query_service_client=query_service_client,
        node=node,  # type: ignore
        include_all_revisions=include_all_revisions,
        show_inactive=show_inactive,
        request_headers=request_headers,
    )


@router.delete(
    "/nodes/{node_name}/materializations/",
    response_model=None,
    name="Deactivate a Materialization for a Node",
)
async def deactivate_node_materializations(
    node_name: str,
    materialization_name: str,
    node_version: str | None = None,
    *,
    session: AsyncSession = Depends(get_session),
    request: Request,
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
    current_user: User = Depends(get_current_user),
    save_history: Callable = Depends(get_save_history),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> JSONResponse:
    """
    Deactivate the node materialization with the provided name.
    Also calls the query service to deactivate the associated scheduled jobs.

    If node_version not provided, it will deactivate the materialization
    for the current version of the node.
    """
    access_checker.add_request_by_node_name(node_name, ResourceAction.WRITE)
    await access_checker.check(on_denied=AccessDenialMode.RAISE)
    request_headers = dict(request.headers)

    # find the node revision to deactivate the materialization for
    node_revision = None
    if node_version:
        stmt = (
            select(NodeRevision)
            .options(
                selectinload(NodeRevision.materializations),
                noload(NodeRevision.created_by),
            )
            .where(NodeRevision.name == node_name, NodeRevision.version == node_version)
        )
        result = await session.execute(stmt)
        node_revision = result.scalars().first()
        if not node_revision:
            raise DJDoesNotExistException(  # pragma: no cover
                f"Node revision with version '{node_version}' not found for node {node_name} .",
            )
    else:
        node = await Node.get_by_name(
            session,
            node_name,
            options=[
                noload(Node.created_by),
                noload(Node.tags),
                joinedload(Node.current).options(
                    selectinload(NodeRevision.materializations),
                    noload(NodeRevision.created_by),
                ),
            ],
        )
        node_revision = node.current  # type: ignore

    # find the materialization to deactivate
    materialization_to_deactivate = None
    for materialization in node_revision.materializations:  # type: ignore
        if materialization.name == materialization_name:
            materialization_to_deactivate = materialization
            break
    if not materialization_to_deactivate:
        raise DJDoesNotExistException(
            f"Materialization with name '{materialization_name}' not found on "
            f"version {node_version} of node {node_name} .",
        )
    elif materialization_to_deactivate.deactivated_at:  # pragma: no cover
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={
                "message": f"Materialization named `{materialization_name}` on node `{node_name}` "
                f"version `{node_revision.version}` has been already inactive.",  # type: ignore
            },
        )
    # do the deactivation
    query_service_client.deactivate_materialization(
        node_name,
        materialization_name,
        node_version=node_revision.version,  # type: ignore
        request_headers=request_headers,
    )
    now = datetime.now(UTC)
    materialization_to_deactivate.deactivated_at = UTCDatetime(
        year=now.year,
        month=now.month,
        day=now.day,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
    )
    session.add(materialization_to_deactivate)
    # save the history event
    await save_history(
        event=History(
            entity_type=EntityType.MATERIALIZATION,
            entity_name=materialization_name,
            node=node_name,
            version=node_revision.version,  # type: ignore
            activity_type=ActivityType.DELETE,
            details={},
            user=current_user.username,
        ),
        session=session,
    )
    await session.commit()
    # await session.refresh(node.current)  # type: ignore
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "message": f"Materialization named `{materialization_name}` on node `{node_name}` "
            f"version `{node_revision.version}` has been successfully deactivated",
        },
    )


async def _load_backfill_target(
    session: AsyncSession,
    node_name: str,
    materialization_name: str,
):
    """Load the current node revision, its partition columns and materialization."""
    node = await Node.get_by_name(
        session,
        node_name,
        options=[
            joinedload(Node.current).options(
                selectinload(NodeRevision.columns).options(
                    selectinload(Column.attributes).joinedload(
                        ColumnAttribute.attribute_type,
                    ),
                    selectinload(Column.dimension),
                    selectinload(Column.partition),
                ),
                selectinload(NodeRevision.materializations).selectinload(
                    Materialization.backfills,
                ),
                selectinload(NodeRevision.availability),
            ),
        ],
        raise_if_not_exists=True,
    )
    node_revision = node.current  # type: ignore
    materializations = [
        mat
        for mat in node_revision.materializations
        if mat.name == materialization_name
    ]
    if not materializations:
        raise DJDoesNotExistException(
            f"Materialization with name {materialization_name} not found",
        )
    return node, node_revision, materializations[0]


def _normalize_partition_specs(
    node_revision: NodeRevision,
    backfill_partitions: list[PartitionBackfill],
) -> None:
    """Rewrite spec column names to their amenable partition names (in place)."""
    partition_columns = (
        node_revision.temporal_partition_columns()
        + node_revision.categorical_partition_columns()
    )
    for backfill_spec in backfill_partitions:
        partition_column = resolve_column(
            partition_columns,
            backfill_spec.column_name,
            node_revision.name,
            node_revision.type,
        )
        partition_name = (
            partition_column.cube_element_name
            if node_revision.type == NodeType.CUBE
            else partition_column.name
        )
        backfill_spec.column_name = amenable_name(partition_name)


def _backfill_materialization_info(backfill: Backfill) -> MaterializationInfo:
    """Build the externally-visible scheduling result from a recorded run."""
    return MaterializationInfo(
        output_tables=list(backfill.output_tables or []),
        urls=list(backfill.urls or []),
        workflow_names=[backfill.workflow_name] if backfill.workflow_name else [],
    )


@router.post(
    "/nodes/{node_name}/materializations/{materialization_name}/backfill",
    status_code=201,
    name="Kick off a backfill run for a configured materialization",
)
async def run_materialization_backfill(
    node_name: str,
    materialization_name: str,
    backfill_partitions: list[PartitionBackfill],
    *,
    session: AsyncSession = Depends(get_session),
    request: Request,
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
    current_user: User = Depends(get_current_user),
    save_history: Callable = Depends(get_save_history),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> MaterializationInfo:
    """
    Start an idempotent, generation-tracked backfill for a configured
    materialization.

    The run is recorded (with the pinned materialization version/generation,
    time range, partition plan, workflow name and output identifier) before
    the external workflow is scheduled, so a lost response, partial partition
    completion or a crash before availability is published never leads to a
    duplicate workflow. Retrying the same request adopts the recorded run;
    overlapping runs are arbitrated by materialization generation.
    """
    access_checker.add_request_by_node_name(node_name, ResourceAction.WRITE)
    await access_checker.check(on_denied=AccessDenialMode.RAISE)
    request_headers = dict(request.headers)

    _, node_revision, materialization = await _load_backfill_target(
        session,
        node_name,
        materialization_name,
    )
    _normalize_partition_specs(node_revision, backfill_partitions)

    materialization_jobs = {
        cls.__name__: cls for cls in MaterializationJob.__subclasses__()
    }
    clazz = materialization_jobs.get(materialization.job)
    if not clazz:
        raise DJDoesNotExistException(  # pragma: no cover
            f"Materialization job {materialization.job} does not exist",
        )

    spec_dicts = [
        backfill_partition.model_dump() for backfill_partition in backfill_partitions
    ]
    planned_partitions, temporal_range = backfill_ops.expand_plan(
        node_revision,
        spec_dicts,
    )
    idempotency_key = backfill_ops.make_idempotency_key(
        materialization.id,
        materialization.generation,
        backfill_ops.canonical_spec(backfill_partitions),
    )
    workflow_name = materialization.backfill_workflow

    # A retried request for the same pinned (generation, plan) adopts the run
    # it already recorded -- it never schedules a second workflow.
    existing = (
        await session.execute(
            select(Backfill).where(Backfill.idempotency_key == idempotency_key),
        )
    ).scalar_one_or_none()

    if existing is not None and existing.state != BackfillStatus.FAILED:
        # COMPLETE/SUPERSEDED runs answer idempotently; active runs reconcile
        # with the workflow service. Neither schedules anything new.
        if existing.state.is_active:
            await _reconcile_active_run(
                session,
                existing,
                materialization,
                query_service_client,
                request_headers,
            )
        return _backfill_materialization_info(existing)

    # Arbitrate overlapping ranges by generation. Same-generation overlaps are
    # refused; older-generation runs are superseded before the new run starts.
    superseded = False
    for other in materialization.backfills:
        if not other.state.is_active or other.id == (existing.id if existing else None):
            continue
        if backfill_ops.specs_overlap(
            node_revision,
            other.spec or [],
            spec_dicts,
        ):
            if (other.generation or 1) < (materialization.generation or 1):
                other.status = BackfillStatus.SUPERSEDED.value
                superseded = True
            else:
                raise DJBackfillConflictException(
                    message=(
                        f"An active backfill {other.id} already covers an overlapping"
                        f" range for materialization `{materialization_name}` at"
                        f" generation {materialization.generation}. Wait for it to"
                        " finish or choose a non-overlapping range."
                    ),
                )
    if superseded:
        await session.commit()

    if existing is not None and existing.state == BackfillStatus.FAILED:
        # The previous attempt was rejected before anything ran; the same
        # request is a legitimate retry of that record.
        backfill = existing
        backfill.status = BackfillStatus.PENDING.value
        backfill.confirmed_partitions = []
        backfill.failed_partitions = []
        backfill.published_partitions = []
    else:
        backfill = Backfill(
            materialization=materialization,
            spec=spec_dicts,
            urls=[],
            idempotency_key=idempotency_key,
            materialization_version=node_revision.version,
            generation=materialization.generation,
            workflow_name=workflow_name,
            temporal_range=temporal_range,
            planned_partitions=planned_partitions,
            status=BackfillStatus.PENDING.value,
        )
        materialization.backfills.append(backfill)
        session.add(backfill)

    # Record before scheduling: a crash here leaves a PENDING run recovery can
    # find and reconcile rather than a second workflow.
    await save_history(
        event=History(
            entity_type=EntityType.BACKFILL,
            node=node_name,
            activity_type=ActivityType.CREATE,
            details={
                "materialization": materialization_name,
                "generation": materialization.generation,
                "idempotency_key": idempotency_key,
                "partition": spec_dicts,
            },
            user=current_user.username,
        ),
        session=session,
    )
    await session.commit()
    await session.refresh(backfill)

    materialization_output = await _schedule_backfill(
        backfill,
        materialization,
        backfill_partitions,
        clazz,
        query_service_client,
        request_headers,
    )
    await session.commit()
    return materialization_output


async def _schedule_backfill(
    backfill: Backfill,
    materialization: Materialization,
    backfill_partitions: list[PartitionBackfill],
    clazz: type[MaterializationJob],
    query_service_client: QueryServiceClient,
    request_headers: dict[str, str] | None,
    *,
    allow_resubmit: bool = True,
    reconcile: bool = True,
) -> MaterializationInfo:
    """
    Schedule the external workflow, reconciling an uncertain scheduling call
    by querying the original workflow before any resubmit.

    Recovery passes ``reconcile=False`` because it performed the workflow
    lookup itself before deciding to resubmit; the uncertainty then propagates
    for the endpoint to record and retry on a later recovery.
    """
    try:
        output = clazz().run_backfill(
            materialization,
            backfill_partitions,
            query_service_client,
            workflow_name=backfill.workflow_name,
            idempotency_key=backfill.idempotency_key,
            request_headers=request_headers,
        )
    except DJBackfillScheduleUncertainException:
        if not reconcile:
            raise
        status, resubmitted = await backfill_ops.reconcile_scheduling(
            backfill,
            query_service_client,
            materialization,
            allow_resubmit=allow_resubmit,
            request_headers=request_headers,
        )
        if status is not None:
            output = MaterializationInfo(
                output_tables=list(status.output_tables),
                urls=list(status.urls),
                workflow_names=[status.workflow_name],
            )
        elif resubmitted is not None:
            output = resubmitted
        else:
            # No recorded workflow name means the workflow service cannot be
            # asked whether the run exists; leave the run PENDING for an
            # explicit recovery and report the legacy empty scheduling result.
            _logger.warning(
                "Backfill %s scheduling is uncertain and cannot be reconciled"
                " without a workflow name; left PENDING for recovery.",
                backfill.id,
            )
            return MaterializationInfo(output_tables=[], urls=[])

    # An explicit rejection (for example a materialization with no backfill
    # workflow) keeps the historical empty-result contract; the run simply has
    # nothing running.
    if not output.urls and not output.output_tables and not output.workflow_names:
        backfill.status = BackfillStatus.FAILED.value
        return output

    backfill_ops.adopt_workflow_info(backfill, output)
    backfill.status = BackfillStatus.RUNNING.value
    return _backfill_materialization_info(backfill)


async def _reconcile_active_run(
    session: AsyncSession,
    backfill: Backfill,
    materialization: Materialization,
    query_service_client: QueryServiceClient,
    request_headers: dict[str, str] | None,
) -> None:
    """
    Bring an existing active run up to date with the workflow service when a
    duplicate request arrives. Never resubmits a run that got past scheduling.
    """
    if not backfill.workflow_name:
        return
    status = query_service_client.get_workflow_status(
        backfill.workflow_name,
        request_headers=request_headers,
    )
    if status is None and backfill.state == BackfillStatus.PENDING:
        # Scheduling was lost before the first answer; one resubmit is safe
        # because the workflow service does not know the name.
        partitions = [PartitionBackfill(**entry) for entry in backfill.spec or []]
        materialization_jobs = {
            cls.__name__: cls for cls in MaterializationJob.__subclasses__()
        }
        clazz = materialization_jobs[materialization.job]
        await _schedule_backfill(
            backfill,
            materialization,
            partitions,
            clazz,
            query_service_client,
            request_headers,
            allow_resubmit=True,
        )
    else:
        backfill_ops.adopt_workflow_progress(backfill, status)
        backfill.status = backfill_ops.finalize_status(
            backfill,
            status.state,
        ).value
    await session.commit()


@router.get(
    "/nodes/{node_name}/materializations/{materialization_name}/backfills/",
    response_model=list[BackfillStatusInfo],
    name="List backfill runs for a configured materialization",
)
async def list_materialization_backfills(
    node_name: str,
    materialization_name: str,
    *,
    session: AsyncSession = Depends(get_session),
) -> list[BackfillStatusInfo]:
    """List the generation-tracked backfill runs of a materialization."""
    _, node_revision, materialization = await _load_backfill_target(
        session,
        node_name,
        materialization_name,
    )
    return [
        backfill_ops.to_status_info(
            backfill,
            materialization,
            node_name,
            node_revision.version,
        )
        for backfill in sorted(
            materialization.backfills,
            key=lambda item: item.id,
        )
    ]


@router.get(
    "/nodes/{node_name}/materializations/{materialization_name}/backfills/{backfill_id}",
    response_model=BackfillStatusInfo,
    name="Get a backfill run's generation-tracked status",
)
async def get_materialization_backfill(
    node_name: str,
    materialization_name: str,
    backfill_id: int,
    *,
    session: AsyncSession = Depends(get_session),
) -> BackfillStatusInfo:
    """Get one generation-tracked backfill run."""
    _, node_revision, materialization = await _load_backfill_target(
        session,
        node_name,
        materialization_name,
    )
    backfill = await backfill_ops.get_backfill(
        session,
        backfill_id,
        materialization.id,
    )
    return backfill_ops.to_status_info(
        backfill,
        materialization,
        node_name,
        node_revision.version,
    )


@router.post(
    "/nodes/{node_name}/materializations/{materialization_name}/backfills/{backfill_id}"
    "/completions",
    response_model=BackfillStatusInfo,
    name="Confirm partition outcomes for a backfill run",
)
async def confirm_materialization_backfill_partitions(
    node_name: str,
    materialization_name: str,
    backfill_id: int,
    data: BackfillCompletionInput,
    *,
    session: AsyncSession = Depends(get_session),
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
    current_user: User = Depends(get_current_user),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> BackfillStatusInfo:
    """
    Confirm per-partition outcomes and publish availability for exactly the
    confirmed partitions.

    A reported failure only narrows what gets published -- a run with a failed
    partition can never publish availability covering the whole requested
    range. Confirmations are committed before availability is published, so a
    failed availability write is later repaired as a publish only: completed
    partitions are never recomputed or rerun.
    """
    access_checker.add_request_by_node_name(node_name, ResourceAction.WRITE)
    await access_checker.check(on_denied=AccessDenialMode.RAISE)

    _, node_revision, materialization = await _load_backfill_target(
        session,
        node_name,
        materialization_name,
    )
    backfill = await backfill_ops.get_backfill(
        session,
        backfill_id,
        materialization.id,
    )
    backfill_ops.ensure_generation_current(backfill, materialization)
    if backfill.state in (
        BackfillStatus.FAILED,
        BackfillStatus.SUPERSEDED,
        BackfillStatus.COMPLETE,
    ):
        raise DJBackfillConflictException(
            message=(
                f"Backfill {backfill.id} is {backfill.status}; its partition "
                "outcomes can no longer change."
            ),
        )

    backfill_ops.record_completions(
        node_revision,
        backfill,
        data.partitions,
    )
    # Durable confirmations first: anything that fails below is repaired by
    # republishing, never by recomputing these partitions.
    await session.commit()
    await session.refresh(node_revision, ["availability"])

    await backfill_ops.publish_due_partitions(
        session,
        node_revision,
        backfill,
        data.partitions,
    )
    backfill.status = backfill_ops.finalize_status(backfill).value
    await session.commit()

    return backfill_ops.to_status_info(
        backfill,
        materialization,
        node_name,
        node_revision.version,
    )


@router.post(
    "/nodes/{node_name}/materializations/{materialization_name}/backfills/{backfill_id}"
    "/recover",
    response_model=BackfillStatusInfo,
    name="Recover a backfill run after a failure or service restart",
)
async def recover_materialization_backfill(
    node_name: str,
    materialization_name: str,
    backfill_id: int,
    *,
    session: AsyncSession = Depends(get_session),
    request: Request,
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
    current_user: User = Depends(get_current_user),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> BackfillStatusInfo:
    """
    Resume a run from its three sources of truth -- the pinned Backfill row,
    the external workflow's state and the node revision's availability
    records:

    * a PENDING run the workflow service does not know is scheduled once;
    * partition outcomes the workflow reports are adopted;
    * confirmed partitions already present in availability records are not
      republished, and confirmed-but-unpublished ones are published without
      recomputation;
    * planned partitions the workflow cannot account for stay unpublished and
      are marked failed rather than ever covering the whole range.
    """
    access_checker.add_request_by_node_name(node_name, ResourceAction.WRITE)
    await access_checker.check(on_denied=AccessDenialMode.RAISE)
    request_headers = dict(request.headers)

    _, node_revision, materialization = await _load_backfill_target(
        session,
        node_name,
        materialization_name,
    )
    backfill = await backfill_ops.get_backfill(
        session,
        backfill_id,
        materialization.id,
    )
    backfill_ops.ensure_generation_current(backfill, materialization)

    workflow_status: WorkflowStatusInfo | None = None
    if backfill.workflow_name:
        workflow_status = query_service_client.get_workflow_status(
            backfill.workflow_name,
            request_headers=request_headers,
        )

    if workflow_status is not None:
        backfill_ops.adopt_workflow_progress(backfill, workflow_status)
    elif backfill.state == BackfillStatus.PENDING:
        # Nothing could have run or been published; resubmit exactly once.
        partitions = [PartitionBackfill(**entry) for entry in backfill.spec or []]
        materialization_jobs = {
            cls.__name__: cls for cls in MaterializationJob.__subclasses__()
        }
        clazz = materialization_jobs[materialization.job]
        try:
            await _schedule_backfill(
                backfill,
                materialization,
                partitions,
                clazz,
                query_service_client,
                request_headers,
                reconcile=False,
            )
        except DJBackfillScheduleUncertainException:
            # Still no answer; a later recovery retries the lookup.
            _logger.warning(
                "Backfill %s recovery could not confirm scheduling",
                backfill.id,
            )
    elif backfill.state.is_active:
        # The workflow disappeared mid-run. Unaccounted-for planned partitions
        # cannot be confirmed and must not be silently covered: mark them
        # failed, keep whatever was confirmed. Finalize against a FAILED
        # terminal state so a run that lost its whole workflow does not hang
        # in RUNNING.
        planned = set(backfill.planned_partitions or [])
        confirmed = set(backfill.confirmed_partitions or [])
        failed = set(backfill.failed_partitions or [])
        failed.update(planned - confirmed - failed)
        backfill.failed_partitions = sorted(failed)
        workflow_status = WorkflowStatusInfo(
            workflow_name=backfill.workflow_name or "",
            state=WorkflowRunState.FAILED,
        )

    await session.commit()
    await session.refresh(node_revision, ["availability"])

    # Availability already on the revision counts as published even if the
    # marker on the backfill row was lost in the crash.
    recorded_published = set(backfill.published_partitions or [])
    for key in set(backfill.confirmed_partitions or []) - recorded_published:
        if backfill_ops.availability_covers_key(
            node_revision.availability,
            key,
            node_revision,
        ):
            recorded_published.add(key)
    backfill.published_partitions = sorted(recorded_published)

    # Publish only confirmed partitions the availability records are missing.
    await backfill_ops.publish_due_partitions(
        session,
        node_revision,
        backfill,
    )
    backfill.status = backfill_ops.finalize_status(
        backfill,
        workflow_status.state if workflow_status is not None else None,
    ).value
    await session.commit()

    return backfill_ops.to_status_info(
        backfill,
        materialization,
        node_name,
        node_revision.version,
    )


@router.get(
    "/nodes/{node_name}/availability/",
    response_model=list[AvailabilityStateInfo],
    status_code=200,
    name="List All Availability States for a Node",
)
async def list_node_availability_states(
    node_name: str,
    *,
    session: AsyncSession = Depends(get_session),
) -> list[AvailabilityStateInfo]:
    """
    Retrieve all availability states for a given node across all revisions.
    """
    # Get all revisions with their availability states
    node = await Node.get_by_name(
        session,
        node_name,
        options=[
            joinedload(Node.revisions).options(
                selectinload(NodeRevision.availability),
            ),
        ],
        raise_if_not_exists=True,
    )

    # Collect availability states from all revisions
    availability_states = []
    for revision in node.revisions:  # type: ignore
        if revision.availability:
            availability_state = AvailabilityStateInfo(
                id=revision.availability.id,
                catalog=revision.availability.catalog,
                schema_=revision.availability.schema_,
                table=revision.availability.table,
                valid_through_ts=revision.availability.valid_through_ts,
                url=revision.availability.url,
                links=revision.availability.links,
                categorical_partitions=revision.availability.categorical_partitions,
                temporal_partitions=revision.availability.temporal_partitions,
                min_temporal_partition=revision.availability.min_temporal_partition,
                max_temporal_partition=revision.availability.max_temporal_partition,
                partitions=revision.availability.partitions,
                updated_at=revision.availability.updated_at.isoformat(),
                node_revision_id=revision.id,
                node_version=revision.version,
                total_size_bytes=revision.availability.total_size_bytes,
                total_row_count=revision.availability.total_row_count,
                total_partitions=revision.availability.total_partitions,
                ttl_days=revision.availability.ttl_days,
            )
            availability_states.append(availability_state)

    return availability_states
