"""
Time filter parser for natural language date queries.

parse_time_filter("what was I working on last week?")
  → ("what was I working on", datetime(7 days ago), datetime(now))
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Regex → (date_after_fn, date_before_fn, strip_pattern)
# Evaluated in order; first match wins.
def _rules(now: datetime) -> list[tuple]:
    sod = now.replace(hour=0, minute=0, second=0, microsecond=0)  # start of day
    mon = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    som = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    bow = sod - timedelta(days=now.weekday())  # beginning of week (Monday)

    return [
        (r"\b(last|past)\s+week\b",   now - timedelta(days=7),  now),
        (r"\b(last|past)\s+month\b",  now - timedelta(days=30), now),
        (r"\b(last|past)\s+year\b",   now - timedelta(days=365), now),
        (r"\byesterday\b",            sod - timedelta(days=1),   sod),
        (r"\btoday\b",                sod,                       now),
        (r"\bthis\s+week\b",          bow,                       now),
        (r"\bthis\s+month\b",         som,                       now),
        (r"\bthis\s+year\b",          mon,                       now),
    ]


def parse_time_filter(
    query: str,
) -> tuple[str, Optional[datetime], Optional[datetime]]:
    """
    Returns (cleaned_query, date_after, date_before).
    date_after / date_before are None if no time phrase was detected.
    Strips the matched phrase from the query so the embedding isn't confused.
    """
    now = datetime.now(timezone.utc)
    q_lower = query.lower()

    # Fixed-phrase rules
    for pattern, date_after, date_before in _rules(now):
        if re.search(pattern, q_lower):
            cleaned = re.sub(pattern, "", query, flags=re.IGNORECASE).strip(" ,?.")
            return cleaned or query, date_after, date_before

    # "N days/weeks/months ago"
    m = re.search(r"\b(\d+)\s+(day|week|month)s?\s+ago\b", q_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(days=n if unit == "day" else n * 7 if unit == "week" else n * 30)
        cleaned = re.sub(r"\b\d+\s+(day|week|month)s?\s+ago\b", "", query, flags=re.IGNORECASE).strip(" ,?.")
        return cleaned or query, now - delta, now

    # "in [month name]"
    m = re.search(
        r"\bin\s+(january|february|march|april|may|june|july|august"
        r"|september|october|november|december)\b",
        q_lower,
    )
    if m:
        month_num = _MONTHS[m.group(1)]
        year = now.year
        if month_num > now.month:
            year -= 1  # "in March" when it's April → last March
        date_after = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            date_before = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            date_before = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        cleaned = re.sub(r"\bin\s+\w+\b", "", query, flags=re.IGNORECASE).strip(" ,?.")
        return cleaned or query, date_after, date_before

    return query, None, None
