from datetime import UTC, datetime
from functools import partial
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_utils import UUIDType

from datajunction_server.database.base import Base
from datajunction_server.database.user import User
from datajunction_server.errors import DJError, ErrorCode
from datajunction_server.models.deployment import (
    DeploymentProgress,
    DeploymentResult,
    DeploymentSnapshot,
    NodeCheckResults,
    DeploymentSpec,
    DeploymentStatus,
)
from datajunction_server.models.impact import DownstreamImpact
from datajunction_server.typing import UTCDatetime


class Deployment(Base):
    __tablename__ = "deployments"

    __table_args__ = (
        # For queries filtering by namespace + status, ordered by created_at
        Index(
            "ix_deployments_namespace_status_created",
            "namespace",
            "status",
            "created_at",
        ),
        # For queries filtering by just namespace
        Index("ix_deployments_namespace", "namespace"),
    )

    uuid: Mapped[UUID] = mapped_column(UUIDType(), default=uuid4, primary_key=True)

    namespace: Mapped[str] = mapped_column(String)
    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus),
        default=DeploymentStatus.PENDING,
    )
    spec: Mapped[dict] = mapped_column(JSON, default={})
    results: Mapped[dict] = mapped_column(JSON, default={})
    warnings: Mapped[list | None] = mapped_column(JSON, default=list)
    downstream_impacts: Mapped[list | None] = mapped_column(JSON, default=list)
    check_results: Mapped[list | None] = mapped_column(JSON, default=list)

    # Client-supplied idempotency key (Idempotency-Key header). Nullable;
    # NULLs never collide, so keyless creates always get a fresh deployment.
    idempotency_key: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        unique=True,
    )
    # Immutable pin for git deployments to managed namespaces; None for local
    # deployments, dry-run impact previews, and rows written before pinning.
    snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    # Recoverable per-stage progress (lock/authz/nodes); None for legacy rows.
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    # Fencing token of the managed-namespace lock this deployment currently
    # holds (None when it holds no lock).
    lock_token: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    started_at: Mapped[UTCDatetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    finished_at: Mapped[UTCDatetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_by: Mapped[User] = relationship("User", lazy="selectin")

    created_at: Mapped[UTCDatetime] = mapped_column(
        DateTime(timezone=True),
        default=partial(datetime.now, UTC),
    )
    updated_at: Mapped[UTCDatetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    @property
    def deployment_spec(self) -> DeploymentSpec:
        return DeploymentSpec(**self.spec)  # pragma: no cover

    @deployment_spec.setter
    def deployment_spec(self, value: DeploymentSpec):
        self.spec = value.model_dump()  # pragma: no cover

    @property
    def deployment_results(self) -> list[DeploymentResult]:
        return [DeploymentResult(**result) for result in self.results]

    @deployment_results.setter
    def deployment_results(self, value: list[DeploymentResult]):
        self.results = [result.model_dump() for result in value]  # pragma: no cover

    @property
    def deployment_check_results(self) -> list[NodeCheckResults]:
        return [NodeCheckResults(**item) for item in (self.check_results or [])]

    @deployment_check_results.setter
    def deployment_check_results(self, value: list[NodeCheckResults]):
        self.check_results = [item.model_dump() for item in value]

    @property
    def deployment_warnings(self) -> list[DJError]:
        # ``code`` is stored as the symbolic ErrorCode name (see
        # DJError.serialize_code), so resolve it back to the enum member.
        return [
            DJError(**{**warning, "code": ErrorCode[warning["code"]]})
            for warning in (self.warnings or [])
        ]

    @deployment_warnings.setter
    def deployment_warnings(self, value: list[DJError]):
        self.warnings = [warning.model_dump() for warning in value]

    @property
    def deployment_downstream_impacts(self) -> list[DownstreamImpact]:
        return [DownstreamImpact(**item) for item in (self.downstream_impacts or [])]

    @property
    def deployment_snapshot(self) -> DeploymentSnapshot | None:
        return DeploymentSnapshot(**self.snapshot) if self.snapshot else None

    @deployment_snapshot.setter
    def deployment_snapshot(self, value: DeploymentSnapshot | None):
        self.snapshot = value.model_dump(mode="json") if value else None

    @property
    def deployment_progress(self) -> DeploymentProgress | None:
        if not self.progress:
            return None
        return DeploymentProgress(**self.progress)

    @deployment_progress.setter
    def deployment_progress(self, value: DeploymentProgress | None):
        self.progress = value.model_dump(mode="json") if value else None

    @property
    def is_pinned(self) -> bool:
        """Whether this deployment carries the immutable git snapshot."""
        return bool(self.snapshot)

    @property
    def is_terminal(self) -> bool:
        return self.status in DeploymentStatus.terminal()
