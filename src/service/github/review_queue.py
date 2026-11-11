import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial

from src.service.deepseek_price import is_off_peak
from src.service.github.models import (
    RepoContainerState,
    ReviewConfig,
    ReviewHistory,
    ReviewQueueItem,
    ReviewReport,
    ReviewResult,
)
from src.service.github.pr_container import PullRequestContainer
from src.service.github.review import review_pr
from src.service.github.review_comments import update_review_comment

import nonebot
from nonebot.adapters.github import GitHubBot
from nonebot.log import logger
from nonebot_plugin_orm import get_session
from pydantic_ai.run import AgentRun
from sqlalchemy import delete, select, update

config = nonebot.get_plugin_config(ReviewConfig)


@dataclass
class ReviewRequest:
    bot: GitHubBot
    owner: str
    repo: str
    pr: int
    installation_id: int
    callback: Callable[..., Awaitable[None]]
    extra_prompt: str | None = None
    comment_id: int | None = None
    summary_comment_id: int | None = None
    queued_notified: bool = False
    id: int | None = None
    priority: int = 1
    followup: Callable[[str], bool] | None = None
    pending_followup: bool = False
    started: bool = False
    comment_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RepositoryReviews:
    def __init__(self, owner: str, repo: str):
        self.container = PullRequestContainer(owner, repo)
        self.pending: deque[ReviewRequest] = deque()
        self.worker: asyncio.Task | None = None
        self.current: asyncio.Task | None = None
        self.current_pr: int | None = None
        self.current_request: ReviewRequest | None = None
        self.last_used = datetime.now(UTC)
        self.lock = asyncio.Lock()
        self.pending_lock = asyncio.Lock()
        self.start_allowed = asyncio.Event()
        self.stopping = False

    async def submit(self, request: ReviewRequest) -> None:
        logger.info(
            "Enqueue review: repo={}/{} pr={} comment={} app={}",
            request.owner,
            request.repo,
            request.pr,
            request.comment_id,
            request.bot.self_id,
        )
        async with self.pending_lock:
            request.priority = 0 if self.current_pr == request.pr else 1
            async with get_session() as session:
                item = ReviewQueueItem(
                    owner=request.owner,
                    repo=request.repo,
                    pr=request.pr,
                    app_id=request.bot.self_id,
                    installation_id=request.installation_id,
                    comment_id=request.comment_id,
                    extra_prompt=request.extra_prompt,
                    priority=request.priority,
                    status="pending",
                    queued_notified=False,
                )
                session.add(item)
                await session.flush()
                request_id = item.id
                # Insert first, and delete only older versions; a finishing old run must not delete its replacement.
                await session.execute(
                    delete(ReviewQueueItem).where(
                        ReviewQueueItem.owner == request.owner,
                        ReviewQueueItem.repo == request.repo,
                        ReviewQueueItem.pr == request.pr,
                        ReviewQueueItem.id != request_id,
                    )
                )
                await session.commit()
            request.id = request_id
            logger.info(
                "Review persisted: request={} repo={}/{} pr={}", request.id, request.owner, request.repo, request.pr
            )
            replaced = [item for item in self.pending if item.pr == request.pr]
            self.pending = deque(item for item in self.pending if item.pr != request.pr)
            if self.current_pr == request.pr:
                self.pending.appendleft(request)
                self.cancel_current()
            else:
                self.pending.append(request)
            reason = self.waiting_reason(request)
            if reason:
                await self.notify_queued(request, reason)
            self.kick()
        for item in replaced:
            if item.summary_comment_id is not None:
                await self.set_status(
                    item, "## Review cancelled\n\nThis request was replaced by a newer review request."
                )
        if reason:
            logger.info(
                "Review waiting: request={} repo={}/{} pr={} reason={}",
                request.id,
                request.owner,
                request.repo,
                request.pr,
                reason.replace("\n", "; "),
            )

    async def edit_comment(self, pr: int, comment_id: int, extra_prompt: str | None) -> bool:
        message = (
            "The user has modified the additional requirements. Replace the previous additional requirements "
            "with the following:\n"
            + (extra_prompt or "No additional requirements; use the standard review instructions.")
        )
        async with self.pending_lock:
            async with get_session() as session:
                item = await session.scalar(
                    select(ReviewQueueItem).where(
                        ReviewQueueItem.owner == self.container.owner,
                        ReviewQueueItem.repo == self.container.repo,
                        ReviewQueueItem.pr == pr,
                        ReviewQueueItem.comment_id == comment_id,
                    )
                )
                if item is None:
                    logger.info(
                        "Ignore edited comment without an active review: repo={}/{} pr={} comment={}",
                        self.container.owner,
                        self.container.repo,
                        pr,
                        comment_id,
                    )
                    return False
                item.extra_prompt = message
                request_id = item.id
                await session.commit()
            for request in [*self.pending, *([self.current_request] if self.current_request is not None else [])]:
                if request.id == request_id:
                    request.extra_prompt = message
                    request.pending_followup = True
                    if request.followup is not None and request.followup(message):
                        request.pending_followup = False
                    break
            logger.info(
                "Review requirements updated: request={} repo={}/{} pr={} comment={}",
                request_id,
                self.container.owner,
                self.container.repo,
                pr,
                comment_id,
            )
            return True

    @staticmethod
    def bind_run(request: ReviewRequest, run: AgentRun[object, ReviewReport]) -> None:
        def followup(message: str) -> bool:
            if run.result is not None:
                return False
            run.enqueue(message, priority="asap")
            logger.info(
                "Follow-up enqueued for Agent: request={} repo={}/{} pr={} comment={}",
                request.id,
                request.owner,
                request.repo,
                request.pr,
                request.comment_id,
            )
            return True

        request.followup = followup
        if request.pending_followup and followup(request.extra_prompt or ""):
            request.pending_followup = False

    def waiting_reason(self, request: ReviewRequest) -> str:
        reasons = []
        if config.review_need_off_peak and not is_off_peak():
            reasons.append("In DeepSeek peak-hours; this review will start during the next off-peak period.")
        if self.current_pr is not None and self.current_pr != request.pr:
            reasons.append(f"PR #{self.current_pr} is currently being reviewed in this repository.")
        elif self.pending and self.pending[0] is not request:
            reasons.append("Other PR reviews are queued ahead in this repository.")
        return "\n".join(f"- {reason}" for reason in reasons)

    async def notify_queued(self, request: ReviewRequest, reason: str) -> None:
        await self.set_status(request, f"## Review queued\n\n{reason}", queued=True)

    async def set_status(self, request: ReviewRequest, body: str, *, queued: bool = False) -> None:
        async with request.comment_lock:
            if queued and (request.queued_notified or request.started):
                return
            try:
                request.summary_comment_id = await update_review_comment(
                    request.bot,
                    owner=request.owner,
                    repo=request.repo,
                    pr=request.pr,
                    installation_id=request.installation_id,
                    comment_id=request.summary_comment_id,
                    body=body,
                )
                if queued:
                    request.queued_notified = True
                async with get_session() as session:
                    await session.execute(
                        update(ReviewQueueItem)
                        .where(ReviewQueueItem.id == request.id)
                        .values(summary_comment_id=request.summary_comment_id, queued_notified=request.queued_notified)
                    )
                    await session.commit()
            except Exception:
                logger.exception("Failed to update review status: request={} pr={}", request.id, request.pr)

    async def complete(self, request: ReviewRequest, result: ReviewResult | Exception, *, head_sha: str) -> None:
        # Queue notices and completion must not race to create or overwrite the same comment.
        async with request.comment_lock:
            await request.callback(result, head_sha=head_sha, summary_comment_id=request.summary_comment_id)

    async def notify_started(self, request: ReviewRequest) -> None:
        request.started = True
        await self.set_status(request, "## Review started\n\nPreparing the review environment and starting the Agent.")

    def cancel_current(self) -> None:
        if self.current is not None and not self.current.done() and not self.current.cancelling():
            logger.info(
                "Cancelling review: repo={}/{} pr={}", self.container.owner, self.container.repo, self.current_pr
            )
            self.current.cancel()

    async def close(self, pr: int, *, comment_id: int | None = None) -> None:
        async with self.pending_lock:
            removed = [
                item for item in self.pending if item.pr == pr and (comment_id is None or item.comment_id == comment_id)
            ]
            async with get_session() as session:
                statement = delete(ReviewQueueItem).where(
                    ReviewQueueItem.owner == self.container.owner,
                    ReviewQueueItem.repo == self.container.repo,
                    ReviewQueueItem.pr == pr,
                )
                if comment_id is not None:
                    statement = statement.where(ReviewQueueItem.comment_id == comment_id)
                await session.execute(statement)
                await session.commit()
            self.pending = deque(
                item
                for item in self.pending
                if not (item.pr == pr and (comment_id is None or item.comment_id == comment_id))
            )
            if self.current_pr == pr and (
                comment_id is None
                or (self.current_request is not None and self.current_request.comment_id == comment_id)
            ):
                self.cancel_current()
            logger.info(
                "Review cancellation processed: repo={}/{} pr={} comment={}",
                self.container.owner,
                self.container.repo,
                pr,
                comment_id,
            )
        for item in removed:
            if item.summary_comment_id is not None:
                await self.set_status(
                    item, "## Review cancelled\n\nThe PR was closed or its review command was deleted."
                )

    def kick(self) -> None:
        if self.stopping:
            return
        if config.review_need_off_peak and not is_off_peak():
            logger.debug(
                "Queue deferred until off-peak: repo={}/{} pending={}",
                self.container.owner,
                self.container.repo,
                len(self.pending),
            )
            self.start_allowed.clear()
            return
        self.start_allowed.set()
        if not self.pending or (self.worker is not None and not self.worker.done()):
            return
        self.worker = asyncio.create_task(self._work())
        logger.info(
            "Review worker started: repo={}/{} pending={}", self.container.owner, self.container.repo, len(self.pending)
        )

    async def wait_for_start(self, request: ReviewRequest) -> None:
        waited = False
        while config.review_need_off_peak and not is_off_peak():
            self.start_allowed.clear()
            if not waited:
                await self.set_status(request, "## Review queued\n\nWaiting for the next off-peak period.")
                waited = True
            if not config.review_need_off_peak or is_off_peak():
                break
            await self.start_allowed.wait()
        if waited:
            await self.notify_started(request)

    async def _work(self) -> None:
        try:
            while self.pending and not self.stopping:
                if config.review_need_off_peak and not is_off_peak():
                    break
                async with self.pending_lock:
                    if not self.pending or self.stopping:
                        break
                    request = self.pending.popleft()
                    async with get_session() as session:
                        claimed = await session.scalar(
                            update(ReviewQueueItem)
                            .where(
                                ReviewQueueItem.id == request.id,
                                ReviewQueueItem.status == "pending",
                            )
                            .values(status="running")
                            .returning(ReviewQueueItem.id)
                        )
                        await session.commit()
                    if claimed is None:
                        continue
                    self.current_pr = request.pr
                    self.current_request = request
                    self.current = asyncio.create_task(self._review(request))
                    logger.info(
                        "Review claimed: request={} repo={}/{} pr={} comment={}",
                        request.id,
                        request.owner,
                        request.repo,
                        request.pr,
                        request.comment_id,
                    )
                cancelled = False
                try:
                    await self.current
                except asyncio.CancelledError:
                    cancelled = True
                    # Cancelling a review is a handover; cancelling the worker is shutdown.
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                except Exception as exc:
                    logger.exception(f"Review failed for {request.owner}/{request.repo}#{request.pr}")
                    try:
                        await self.complete(request, exc, head_sha="")
                    except Exception:
                        logger.exception("Failed to post review failure")
                finally:
                    async with self.pending_lock:
                        async with get_session() as session:
                            if (self.stopping and cancelled) or (request.pending_followup and not cancelled):
                                await session.execute(
                                    update(ReviewQueueItem)
                                    .where(
                                        ReviewQueueItem.id == request.id,
                                    )
                                    .values(status="pending", priority=0)
                                )
                                if not self.stopping:
                                    request.priority = 0
                                    self.pending.appendleft(request)
                                    logger.info("Late requirements edit retained for follow-up: request={}", request.id)
                            else:
                                await session.execute(delete(ReviewQueueItem).where(ReviewQueueItem.id == request.id))
                            await session.commit()
                        self.current = None
                        self.current_pr = None
                        self.current_request = None
        except Exception:
            logger.exception(f"Review queue failed for {self.container.owner}/{self.container.repo}")
        finally:
            self.worker = None

    async def _save_state(self, pr: int | None) -> None:
        async with get_session() as session:
            state = await session.get(RepoContainerState, (self.container.owner, self.container.repo))
            if state is None:
                state = RepoContainerState(owner=self.container.owner, repo=self.container.repo)
                session.add(state)
            state.container_id = self.container.container_id
            state.pr = pr
            state.last_used = self.last_used
            await session.commit()

    async def _review(self, request: ReviewRequest) -> None:
        async with self.lock:
            callback = partial(self.complete, request, head_sha="")
            try:
                logger.info(
                    "Fetching PR: request={} repo={}/{} pr={}", request.id, request.owner, request.repo, request.pr
                )
                # Refresh after queueing, so checkout and inline comments use the same current commit.
                async with request.bot.as_installation(request.installation_id):
                    response = await request.bot.github.rest.pulls.async_get(
                        owner=request.owner, repo=request.repo, pull_number=request.pr
                    )
                pull_request = response.parsed_data
                if pull_request.state != "open":
                    logger.info(
                        "Skipping closed PR: request={} repo={}/{} pr={}",
                        request.id,
                        request.owner,
                        request.repo,
                        request.pr,
                    )
                    if request.summary_comment_id is not None:
                        await self.set_status(request, "## Review cancelled\n\nThis pull request has been closed.")
                    return
                callback = partial(self.complete, request, head_sha=pull_request.head.sha)
                await self.notify_started(request)
                self.last_used = datetime.now(UTC)
                await self._save_state(request.pr)
                logger.info("Preparing review container: request={} name={}", request.id, self.container.container_name)
                await self.container.init(request.pr, head_sha=pull_request.head.sha)
                logger.info("Review container ready: request={} container={}", request.id, self.container.container_id)
                await self._save_state(request.pr)
                while True:
                    request.pending_followup = False
                    logger.info(
                        "Starting Agent review: request={} repo={}/{} pr={}",
                        request.id,
                        request.owner,
                        request.repo,
                        request.pr,
                    )
                    try:
                        async with get_session() as session:
                            await review_pr(
                                session=session,
                                owner=request.owner,
                                repo=request.repo,
                                pr=request.pr,
                                base_sha=pull_request.base.sha,
                                head_sha=pull_request.head.sha,
                                installation_id=request.installation_id,
                                container=self.container,
                                wait_for_start=partial(self.wait_for_start, request),
                                bot=request.bot,
                                pr_title=pull_request.title,
                                pr_description=pull_request.body or "",
                                extra_prompt=request.extra_prompt,
                                callback=callback,
                                on_run_ready=partial(self.bind_run, request),
                            )
                    finally:
                        request.followup = None
                    if not request.pending_followup:
                        break
                    logger.info("Continuing review with requirements edited during completion: request={}", request.id)
            except asyncio.CancelledError:
                body = (
                    "## Review paused\n\nThe service is shutting down; this review will resume after restart."
                    if self.stopping
                    else "## Review cancelled\n\nThis review request was cancelled or replaced."
                )
                await self.set_status(request, body)
                raise
            except Exception as exc:
                logger.exception(
                    "Review failed: request={} repo={}/{} pr={} error_type={}",
                    request.id,
                    request.owner,
                    request.repo,
                    request.pr,
                    type(exc).__name__,
                )
                await callback(exc)
            finally:
                request.followup = None
                self.container.free_job()
                logger.info(
                    "Review PR released; container retained: request={} repo={}/{} pr={} container={}",
                    request.id,
                    request.owner,
                    request.repo,
                    request.pr,
                    self.container.container_id,
                )
                self.last_used = datetime.now(UTC)
                release = asyncio.create_task(self._save_state(None))
                try:
                    await asyncio.shield(release)
                except asyncio.CancelledError:
                    await release
                    raise


class ReviewQueues:
    def __init__(self):
        self.repositories: dict[tuple[str, str], RepositoryReviews] = {}
        self.callback: Callable[..., Awaitable[None]] | None = None
        self.stopping = False
        self.ready = False

    async def submit(self, request: ReviewRequest) -> None:
        if self.stopping:
            raise RuntimeError("Review queues are shutting down")
        key = (request.owner, request.repo)
        if key not in self.repositories:
            self.repositories[key] = RepositoryReviews(*key)
        await self.repositories[key].submit(request)

    async def edit_comment(self, owner: str, repo: str, pr: int, comment_id: int, extra_prompt: str | None) -> None:
        key = (owner, repo)
        queue = self.repositories.setdefault(key, RepositoryReviews(*key))
        await queue.edit_comment(pr, comment_id, extra_prompt)

    async def close(self, owner: str, repo: str, pr: int, *, comment_id: int | None = None) -> None:
        key = (owner, repo)
        queue = self.repositories.setdefault(key, RepositoryReviews(*key))
        await queue.close(pr, comment_id=comment_id)

    async def wake(self) -> None:
        if self.stopping or not self.ready:
            return
        await self._load_pending()
        for queue in list(self.repositories.values()):
            # A restart between inserting a request and sending its notice must not lose the notice.
            for request in list(queue.pending):
                if not request.queued_notified and (reason := queue.waiting_reason(request)):
                    await queue.notify_queued(request, reason)
            queue.kick()

    async def restore(self, callback: Callable[..., Awaitable[None]]) -> None:
        logger.info("Restoring review queues from database")
        self.callback = callback
        async with get_session() as session:
            for state in await session.scalars(select(RepoContainerState)):
                key = (state.owner, state.repo)
                queue = self.repositories.setdefault(key, RepositoryReviews(*key))
                queue.container.container_id = state.container_id
                queue.last_used = state.last_used.replace(tzinfo=UTC)
                state.pr = None
            # A process restart interrupted these runs. Keep their place ahead of waiting PRs.
            await session.execute(
                update(ReviewQueueItem)
                .where(
                    ReviewQueueItem.status == "running",
                )
                .values(status="pending", priority=0)
            )
            await session.commit()
        self.ready = True
        await self.wake()
        logger.info(
            "Review queues restored: repositories={} pending={}",
            len(self.repositories),
            sum(len(queue.pending) for queue in self.repositories.values()),
        )

    async def _load_pending(self) -> None:
        assert self.callback is not None
        async with get_session() as session:
            items = (
                await session.scalars(
                    select(ReviewQueueItem)
                    .where(
                        ReviewQueueItem.status == "pending",
                    )
                    .order_by(ReviewQueueItem.priority, ReviewQueueItem.id)
                )
            ).all()
        for item in items:
            bot = nonebot.get_bots().get(item.app_id)
            if not isinstance(bot, GitHubBot):
                logger.debug("Persisted review waiting for App connection: request={} app={}", item.id, item.app_id)
                # The bot-connect hook (and periodic wake) retries once the matching App is available.
                continue
            key = (item.owner, item.repo)
            queue = self.repositories.setdefault(key, RepositoryReviews(*key))
            async with queue.pending_lock:
                if (queue.current_request is not None and queue.current_request.id == item.id) or any(
                    request.id == item.id for request in queue.pending
                ):
                    continue
                queue.pending.append(
                    ReviewRequest(
                        bot=bot,
                        owner=item.owner,
                        repo=item.repo,
                        pr=item.pr,
                        installation_id=item.installation_id,
                        extra_prompt=item.extra_prompt,
                        comment_id=item.comment_id,
                        summary_comment_id=item.summary_comment_id,
                        queued_notified=item.queued_notified,
                        id=item.id,
                        priority=item.priority,
                        callback=partial(
                            self.callback,
                            bot=bot,
                            owner=item.owner,
                            repo=item.repo,
                            pr=item.pr,
                            installation_id=item.installation_id,
                        ),
                    )
                )
                queue.pending = deque(sorted(queue.pending, key=lambda request: (request.priority, request.id or 0)))

    async def cleanup(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=3)
        for key, queue in list(self.repositories.items()):
            if queue.current_pr is not None or queue.pending or queue.lock.locked():
                continue
            try:
                async with queue.lock:
                    if queue.pending or queue.current_pr is not None or queue.last_used > cutoff:
                        continue
                    async with get_session() as session:
                        waiting = await session.scalar(
                            select(ReviewQueueItem.id)
                            .where(
                                ReviewQueueItem.owner == key[0],
                                ReviewQueueItem.repo == key[1],
                            )
                            .limit(1)
                        )
                        if waiting is not None:
                            continue
                    logger.info("Removing idle repository container: repo={}/{} last_used={}", *key, queue.last_used)
                    await queue.container.free_container()
                    async with get_session() as session:
                        await session.execute(
                            delete(ReviewHistory).where(ReviewHistory.owner == key[0], ReviewHistory.repo == key[1])
                        )
                        await session.execute(
                            delete(RepoContainerState).where(
                                RepoContainerState.owner == key[0], RepoContainerState.repo == key[1]
                            )
                        )
                        await session.commit()
                    if not queue.pending and queue.current_pr is None:
                        self.repositories.pop(key, None)
            except Exception:
                logger.exception(f"Failed to clean up container for {key[0]}/{key[1]}")

    async def shutdown(self) -> None:
        self.stopping = True
        workers = []
        for queue in self.repositories.values():
            queue.stopping = True
            queue.pending.clear()
            queue.cancel_current()
            if queue.worker is not None:
                workers.append(queue.worker)
        await asyncio.gather(*workers, return_exceptions=True)
