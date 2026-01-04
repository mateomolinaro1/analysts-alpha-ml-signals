import polars as pl
from datetime import date, datetime
from dataclasses import dataclass

def _validate_date(dt: date | datetime | str) -> date:
    """
    Validate and convert input date to a valid date object.

    Parameters
    ----------
    date: date, datetime, or string
        Input date to validate

    Returns
    -------
    date
        Validated date object
    """
    if isinstance(dt, datetime):
        return dt.date()
    if isinstance(dt, date):
        return dt
    if isinstance(dt, str):
        try:
            return datetime.strptime(dt, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"String date '{dt}' must be in 'YYYY-MM-DD' format.") from e

    raise TypeError("Date must be of type date, datetime, or string.")

@dataclass
class Timeline:
    warmup: list[date]
    backtest: list[date]
    all: list[date]

def _parse_date(ym: str) -> date:
        y, m = ym.split("-")
        return date(int(y), int(m), 1)

def _add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, 1)

def _month_range(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d = _add_months(d, 1)
    return out