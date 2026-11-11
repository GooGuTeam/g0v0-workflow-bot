from githubkit.exception import RequestFailed
from nonebot.adapters.github import GitHubBot


async def update_review_comment(
    bot: GitHubBot,
    *,
    owner: str,
    repo: str,
    pr: int,
    installation_id: int,
    body: str,
    comment_id: int | None,
) -> int:
    """Update the review's status/summary comment, recreating it if it was deleted."""
    async with bot.as_installation(installation_id):
        if comment_id is not None:
            try:
                response = await bot.github.rest.issues.async_update_comment(
                    owner=owner, repo=repo, comment_id=comment_id, body=body
                )
                return response.parsed_data.id
            except RequestFailed as exc:
                if exc.response.status_code != 404:
                    raise
        response = await bot.github.rest.issues.async_create_comment(owner=owner, repo=repo, issue_number=pr, body=body)
        return response.parsed_data.id
