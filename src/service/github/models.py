from datetime import datetime
from typing import Literal

from nonebot_plugin_orm import Model
from pydantic import BaseModel, Field
from pydantic_ai import RunUsage
from pydantic_ai.settings import ThinkingLevel
from sqlalchemy import BigInteger, Text
from sqlalchemy.orm import Mapped, mapped_column


class ReviewConfig(BaseModel):
    provider: str | None = "deepseek"
    review_base_url: str | None = None
    review_model: str = "deepseek-flash"
    review_api_key: str | None = None
    review_thinking: ThinkingLevel = "high"
    review_repos: list[str] = Field(default_factory=list)
    review_need_off_peak: bool = False
    review_request_limit: int | None = Field(default=1000, gt=0)

    def validate_provider(self) -> None:
        if self.review_base_url is None and self.provider is None:
            raise ValueError("ReviewConfig requires either review_base_url or provider; both are None")


class ReviewHistory(Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[str] = mapped_column(index=True)
    repo: Mapped[str] = mapped_column(index=True)
    pr: Mapped[int] = mapped_column(index=True)
    sha: Mapped[str] = mapped_column()
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    date: Mapped[datetime] = mapped_column()
    conversation: Mapped[str] = mapped_column(Text)


class RepoContainerState(Model):
    owner: Mapped[str] = mapped_column(primary_key=True)
    repo: Mapped[str] = mapped_column(primary_key=True)
    container_id: Mapped[str | None] = mapped_column()
    pr: Mapped[int | None] = mapped_column()
    last_used: Mapped[datetime] = mapped_column()


class ReviewQueueItem(Model):
    __table_args__ = {"sqlite_autoincrement": True}  # noqa: RUF012 - SQLAlchemy declarative table options

    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[str] = mapped_column(index=True)
    repo: Mapped[str] = mapped_column(index=True)
    pr: Mapped[int] = mapped_column(index=True)
    app_id: Mapped[str] = mapped_column()
    installation_id: Mapped[int] = mapped_column(BigInteger)
    comment_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    summary_comment_id: Mapped[int | None] = mapped_column(BigInteger)
    extra_prompt: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(default="pending", index=True)
    priority: Mapped[int] = mapped_column(default=1)
    queued_notified: Mapped[bool] = mapped_column(default=False)


class Finding(BaseModel):
    file: str
    side: Literal["LEFT", "RIGHT"] = "RIGHT"
    line_start: int | None = None
    line_end: int | None = None
    severity: Literal["critical", "high", "medium", "low"]
    description: str
    suggestion: str


class ReviewReport(BaseModel):
    summary: str
    findings: list[Finding] = Field(default_factory=list)


class ReviewResult(BaseModel):
    report: ReviewReport
    model_name: str
    usage: RunUsage
    elapsed_seconds: float
