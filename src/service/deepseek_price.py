from datetime import datetime
from zoneinfo import ZoneInfo


def is_off_peak() -> bool:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return now.weekday() >= 5 or not (9 <= now.hour < 12 or 14 <= now.hour < 18)
