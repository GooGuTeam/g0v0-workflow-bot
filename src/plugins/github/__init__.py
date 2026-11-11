from functools import partial
from html import escape
import re

import nonebot

nonebot.require("nonebot_plugin_orm")
nonebot.require("nonebot_plugin_apscheduler")

from src.service.github.models import ReviewConfig, ReviewResult
from src.service.github.review_comments import update_review_comment
from src.service.github.review_queue import ReviewQueues, ReviewRequest

from githubkit.exception import RequestFailed
from githubkit.utils import UNSET
from nonebot import on_command, on_type
from nonebot.adapters import Bot
from nonebot.adapters.github import (
    GitHubBot,
    IssueCommentCreated,
    IssueCommentDeleted,
    IssueCommentEdited,
    Message,
    PullRequestClosed,
    PullRequestOpened,
    PullRequestReadyForReview,
)
from nonebot.log import logger
from nonebot.params import CommandArg
from nonebot.rule import is_type
from nonebot_plugin_apscheduler import scheduler

pr_opened = on_type(PullRequestOpened, block=True)
pr_ready_for_review = on_type(PullRequestReadyForReview, block=True)
pr_comment = on_command("review", rule=is_type(IssueCommentCreated), force_whitespace=True, block=True)
pr_comment_edited = on_type(IssueCommentEdited, block=True)
pr_comment_deleted = on_type(IssueCommentDeleted, block=True)
pr_closed = on_type((PullRequestClosed), block=True)

config = nonebot.get_plugin_config(ReviewConfig)
review_queues = ReviewQueues()

PRTypes = PullRequestOpened | PullRequestReadyForReview
REVIEW_COMMAND = re.compile(r"/review(?:\s+(.*))?", re.DOTALL)


async def review_callback(
    result: ReviewResult | Exception,
    *,
    bot: GitHubBot,
    owner: str,
    repo: str,
    pr: int,
    head_sha: str,
    installation_id: int,
    summary_comment_id: int | None = None,
):
    if isinstance(result, Exception):
        logger.opt(exception=result).warning(
            "Posting review failure: repo={}/{} pr={} error_type={}", owner, repo, pr, type(result).__name__
        )
        await update_review_comment(
            bot,
            owner=owner,
            repo=repo,
            pr=pr,
            installation_id=installation_id,
            comment_id=summary_comment_id,
            body=f"## Review failed\n\n{type(result).__name__}: {result}",
        )
        return
    report = result.report
    logger.info("Posting review result: repo={}/{} pr={} findings={}", owner, repo, pr, len(report.findings))
    sections = ["## Automated review", report.summary]
    usage = result.usage
    sections.append(
        "---\n"
        f"<sub>Model: {escape(result.model_name)} · Tokens: {usage.total_tokens / 1_000_000:.3f} M "
        f"(cache read: {usage.cache_read_tokens / 1_000_000:.3f} M, "
        f"cache write: {usage.cache_write_tokens / 1_000_000:.3f} M) · "
        f"Input: {usage.input_tokens / 1_000_000:.3f} M · "
        f"Output: {usage.output_tokens / 1_000_000:.3f} M · "
        f"Elapsed: {result.elapsed_seconds:.1f} s · "
        f"Requests: {usage.requests} · Tool calls: {usage.tool_calls}</sub>"
    )
    try:
        await update_review_comment(
            bot,
            owner=owner,
            repo=repo,
            pr=pr,
            installation_id=installation_id,
            comment_id=summary_comment_id,
            body="\n\n".join(sections),
        )
        async with bot.as_installation(installation_id):
            for finding in report.findings:
                line = finding.line_end or finding.line_start
                multiline = finding.line_start is not None and line is not None and finding.line_start < line
                await bot.github.rest.pulls.async_create_review_comment(
                    owner=owner,
                    repo=repo,
                    pull_number=pr,
                    commit_id=head_sha,
                    path=finding.file,
                    body=f"**[{finding.severity.upper()}]**\n\n{finding.description}\n\n{finding.suggestion}",
                    subject_type=UNSET if line is not None else "file",
                    line=line if line is not None else UNSET,
                    side=finding.side if line is not None else UNSET,
                    start_line=finding.line_start if multiline else UNSET,
                    start_side=finding.side if multiline else UNSET,
                )
        logger.info("Review comments posted: repo={}/{} pr={}", owner, repo, pr)
    except Exception as exc:
        logger.exception("Review comment publication failed: repo={}/{} pr={}", owner, repo, pr)
        if isinstance(exc, RequestFailed):
            logger.error("GitHub review response: status={} body={}", exc.response.status_code, exc.response.text)
        await review_callback(
            exc,
            bot=bot,
            owner=owner,
            repo=repo,
            pr=pr,
            head_sha=head_sha,
            installation_id=installation_id,
        )


@nonebot.get_driver().on_startup
async def restore_reviews():
    config.validate_provider()
    logger.info(
        "Review configuration: allowed_repos={} need_off_peak={}", config.review_repos, config.review_need_off_peak
    )
    invalid_repos = [repo for repo in config.review_repos if "/" not in repo]
    if invalid_repos:
        logger.warning("Review whitelist requires owner/repo; invalid entries={}", invalid_repos)
    await review_queues.restore(callback=review_callback)


nonebot.get_driver().on_shutdown(review_queues.shutdown)
scheduler.add_job(review_queues.wake, "interval", minutes=1, id="github_review_queue", replace_existing=True)
scheduler.add_job(review_queues.cleanup, "interval", hours=1, id="github_review_cleanup", replace_existing=True)


@nonebot.get_driver().on_bot_connect
async def resume_reviews(bot: Bot):
    if isinstance(bot, GitHubBot):
        logger.info("GitHub App connected; waking review queues: app={}", bot.self_id)
        await review_queues.wake()


async def enqueue_review(
    bot: GitHubBot,
    owner: str,
    repo: str,
    pr: int,
    installation_id: int,
    extra_prompt: str | None = None,
    comment_id: int | None = None,
) -> None:
    await review_queues.submit(
        ReviewRequest(
            bot=bot,
            owner=owner,
            repo=repo,
            pr=pr,
            installation_id=installation_id,
            extra_prompt=extra_prompt,
            comment_id=comment_id,
            callback=partial(review_callback, bot=bot, owner=owner, repo=repo, pr=pr, installation_id=installation_id),
        )
    )


@pr_opened.handle()
@pr_ready_for_review.handle()
async def handle_pr_opened(bot: GitHubBot, event: PRTypes):
    if event.payload.pull_request.draft:
        logger.info(
            "Skipping automatic draft review: repo={} pr={}",
            event.payload.repository.full_name,
            event.payload.pull_request.number,
        )
        return
    if event.payload.repository.full_name not in config.review_repos:
        logger.info(
            "Skipping automatic review: repo={} reason=repository_not_in_review_repos",
            event.payload.repository.full_name,
        )
        return
    installation = event.payload.installation
    if not installation:
        raise ValueError("Installation ID is missing")
    await enqueue_review(
        bot,
        event.payload.repository.owner.login,
        event.payload.repository.name,
        event.payload.pull_request.number,
        installation.id,
    )


@pr_comment.handle()
async def handle_pr_comment(bot: GitHubBot, event: IssueCommentCreated, args: Message = CommandArg()):
    logger.info(
        "Review command received: repo={} issue={} comment={}",
        event.payload.repository.full_name,
        event.payload.issue.number,
        event.payload.comment.id,
    )
    if not event.payload.issue.pull_request:
        logger.info(
            "Skipping /review: repo={} issue={} reason=not_a_pull_request",
            event.payload.repository.full_name,
            event.payload.issue.number,
        )
        return
    if event.payload.repository.full_name not in config.review_repos:
        logger.warning(
            "Skipping /review: repo={} reason=repository_not_in_review_repos; use full owner/repo",
            event.payload.repository.full_name,
        )
        return
    installation = event.payload.installation
    if not installation:
        raise ValueError("Installation ID is missing")
    await enqueue_review(
        bot,
        event.payload.repository.owner.login,
        event.payload.repository.name,
        event.payload.issue.number,
        installation.id,
        args.extract_plain_text().strip() or None,
        comment_id=event.payload.comment.id,
    )


@pr_comment_edited.handle()
async def handle_pr_comment_edited(event: IssueCommentEdited):
    if not event.payload.issue.pull_request or event.payload.repository.full_name not in config.review_repos:
        return
    command = REVIEW_COMMAND.fullmatch(event.payload.comment.body.strip())
    if command is None:
        logger.debug(
            "Ignoring edited non-review comment: repo={} comment={}",
            event.payload.repository.full_name,
            event.payload.comment.id,
        )
        return
    changes = event.payload.changes.body
    if changes and changes.from_.strip() == event.payload.comment.body.strip():
        return
    await review_queues.edit_comment(
        event.payload.repository.owner.login,
        event.payload.repository.name,
        event.payload.issue.number,
        event.payload.comment.id,
        (command.group(1) or "").strip() or None,
    )


@pr_comment_deleted.handle()
async def handle_pr_comment_deleted(event: IssueCommentDeleted):
    if not event.payload.issue.pull_request or event.payload.repository.full_name not in config.review_repos:
        return
    logger.info(
        "PR comment deleted: repo={} pr={} comment={}",
        event.payload.repository.full_name,
        event.payload.issue.number,
        event.payload.comment.id,
    )
    await review_queues.close(
        event.payload.repository.owner.login,
        event.payload.repository.name,
        event.payload.issue.number,
        comment_id=event.payload.comment.id,
    )


@pr_closed.handle()
async def handle_pr_closed(event: PullRequestClosed):
    await review_queues.close(
        event.payload.repository.owner.login, event.payload.repository.name, event.payload.pull_request.number
    )
