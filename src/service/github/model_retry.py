import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
import json
import math
import re
from zoneinfo import ZoneInfo

from nonebot.log import logger
from pydantic_ai.exceptions import ModelHTTPError, ModelQuotaExceededError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

MAX_RETRIES = 5
PLAN_LIMIT = re.compile(
    r"\bplan\b|plan_limit|quota|daily.?limit|weekly.?limit|monthly.?limit|usage.?limit|限额|配额|额度|套餐",
    re.IGNORECASE,
)
DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)", re.IGNORECASE
)
FAST_LIMIT = re.compile(r"per[ _-]?(?:second|minute)|\b(?:qps|rpm|tpm)\b|concurren|每秒|每分钟|并发", re.IGNORECASE)


def _fields(value: object) -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower().replace("-", "_"), child
            yield from _fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from _fields(child)


def _duration(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        seconds = float(str(value))
    except ValueError:
        factors = {"ms": 0.001, "m": 60, "h": 3600, "d": 86400, "s": 1}
        matches = DURATION.findall(str(value))
        if not matches:
            return None
        seconds = sum(
            float(number)
            * (
                factors["ms"]
                if unit.lower().startswith("ms") or unit.lower().startswith("millisecond")
                else factors[unit.lower()[0]]
            )
            for number, unit in matches
        )
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _after(now: datetime, seconds: float | None) -> datetime | None:
    if seconds is None:
        return None
    try:
        return now + timedelta(seconds=seconds)
    except OverflowError:
        return None


def _reset_time(value: object, now: datetime) -> datetime | None:
    try:
        number = float(str(value))
        if not math.isfinite(number):
            return None
        if number >= 1_000_000_000_000:
            number /= 1000
        return datetime.fromtimestamp(number, UTC) if number >= 1_000_000_000 else _after(now, number)
    except (ValueError, OverflowError, OSError):
        pass
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(str(value))
        except (ValueError, TypeError):
            seconds = _duration(value)
            return _after(now, seconds)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def quota_reset(exc: Exception, now: datetime) -> tuple[bool, datetime | None]:
    """Recognize plan limits and provider reset hints without guessing a daily/monthly reset schedule."""
    body = getattr(exc, "body", None)
    if isinstance(body, str):
        with suppress(ValueError):
            body = json.loads(body)
    text = str(body or exc)
    is_plan = isinstance(exc, ModelQuotaExceededError) or PLAN_LIMIT.search(text) is not None
    if FAST_LIMIT.search(text) and not re.search(r"\bplan\b|daily|weekly|monthly|套餐", text, re.IGNORECASE):
        is_plan = False
    candidates: list[datetime] = []
    retry_after = getattr(exc, "retry_after", None)
    if (reset := _after(now, _duration(retry_after))) is not None and reset > now:
        candidates.append(reset)
    headers = getattr(exc, "headers", None) or {}
    for key, value in [*_fields(body), *headers.items()]:
        key = str(key).lower().replace("-", "_")
        if key in {"retry_after", "retry_after_seconds", "reset_in", "reset_after", "reset_in_seconds", "wait_seconds"}:
            seconds = _duration(value)
            reset = _after(now, seconds) if seconds is not None else _reset_time(value, now)
        elif "reset" in key or key in {"retry_at", "available_at"}:
            reset = _reset_time(value, now)
        else:
            continue
        if reset is not None and reset > now:
            candidates.append(reset)
    # Some gateways put their only reset hint in a human-readable error message.
    for value in re.findall(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d+)?)?(?:Z|[+-]\d\d:\d\d)?", text):
        reset = _reset_time(value, now)
        if reset is not None and reset > now:
            candidates.append(reset)
    for value in re.findall(r"(?:retry|try again|resets?|available)\s+(?:after|in)\s+([^,;\n]+)", text, re.IGNORECASE):
        seconds = _duration(value)
        if (reset := _after(now, seconds)) is not None and reset > now:
            candidates.append(reset)
    return is_plan, max(candidates) if candidates else None


class ReviewRequestFailedError(RuntimeError):
    pass


class RetryingReviewModel(WrapperModel):
    def __init__(self, wrapped: Model, notify: Callable[[str], Awaitable[None]]):
        super().__init__(wrapped)
        self.notify = notify
        self.announced_limits: set[str] = set()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        retries = 0
        while True:
            try:
                logger.debug("Model request started: model={} retry={}", self.model_name, retries)
                response = await self.wrapped.request(messages, model_settings, model_request_parameters)
                logger.debug(
                    "Model request completed: model={} input_tokens={} output_tokens={}",
                    self.model_name,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
                return response
            except Exception as exc:
                status = exc.status_code if isinstance(exc, ModelHTTPError) else None
                reason = f"{type(exc).__name__}: {exc}"
                if status in {402, 403, 404}:
                    logger.exception(
                        "Model request stopped: model={} status={} reason=non_retryable_http_error",
                        self.model_name,
                        status,
                    )
                    raise ReviewRequestFailedError(f"Review stopped (HTTP {status}): {reason}") from exc
                now = datetime.now(UTC)
                plan, reset = quota_reset(exc, now)
                limited = status == 429 or isinstance(exc, ModelQuotaExceededError)
                if limited and plan:
                    key = reset.isoformat() if reset is not None else "unknown-reset"
                    if key not in self.announced_limits:
                        self.announced_limits.add(key)
                        target = (
                            f"Retry at {reset.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')} "
                            "(Beijing time)."
                            if reset is not None
                            else "The provider supplied no reset time; retrying after a short delay."
                        )
                        try:
                            await self.notify(f"## Review paused: plan quota reached\n\n{target}\n\n{reason}")
                        except Exception:
                            logger.exception("Failed to post plan quota notification")
                    if reset is not None:
                        logger.opt(exception=True).warning(
                            "Model paused for plan quota: model={} retry_at={}", self.model_name, reset
                        )
                        await asyncio.sleep(max(1, (reset - datetime.now(UTC)).total_seconds()))
                        continue
                if retries >= MAX_RETRIES:
                    logger.exception(
                        "Model request retries exhausted: model={} status={} retries={} error_type={}",
                        self.model_name,
                        status,
                        retries,
                        type(exc).__name__,
                    )
                    raise ReviewRequestFailedError(f"Review stopped after {MAX_RETRIES} retries: {reason}") from exc
                delay = 5.0 if limited else float(2 ** (retries + 1))
                if reset is not None:
                    delay = max(delay, (reset - now).total_seconds())
                retries += 1
                logger.opt(exception=True).warning(
                    "Retrying model request: model={} status={} retry={}/{} delay_seconds={:.1f} error_type={}",
                    self.model_name,
                    status,
                    retries,
                    MAX_RETRIES,
                    delay,
                    type(exc).__name__,
                )
                await asyncio.sleep(delay)
