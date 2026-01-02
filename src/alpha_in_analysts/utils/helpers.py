import polars as pl
from datetime import date, datetime


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
