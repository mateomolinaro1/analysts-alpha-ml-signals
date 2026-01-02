from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "logger.log"
DATA_PATH = PROJECT_ROOT / "data"

# At least, the query must retrieve the following columns:
# ['ticker','exchcd','cusip','ncusip','comnam','permno','permco','namedt','nameendt','date']
STARTING_DATE = "1999-01-01" # of the wrds query YYYY-MM-DD
ENDING_DATE = "2009-12-31"
WRDS_REQUEST = """
SELECT
    a.ticker, a.exchcd,
    a.comnam, a.cusip, a.ncusip,
    a.permno, a.permco,
    a.namedt, a.nameendt,
    b.date, b.ret, b.prc, b.shrout, b.vol,
    ABS(b.prc) * b.shrout * 1000 AS market_cap
FROM crsp.msenames AS a
JOIN crsp.dsf AS b
  ON a.permno = b.permno
 AND b.date BETWEEN a.namedt AND a.nameendt
WHERE a.exchcd IN (1, 2, 3)          -- NYSE, AMEX, NASDAQ
  AND a.shrcd IN (10, 11)            -- Common shares only
  AND b.date >= '{starting_date}'
  AND b.date <= '{ending_date}'
  AND b.prc IS NOT NULL              -- ensure valid price
  AND b.vol IS NOT NULL              -- ensure valid volume
  AND b.prc != 0                     -- avoid zero-price issues
ORDER BY b.date
"""

DATE_COLS = [
    'namedt',
    'nameendt',
    'date'
]

SAVING_CONFIG_UNIVERSE = {
    'gross_query': {
        'path': DATA_PATH / "wrds_gross_query1.parquet",
        'extension': 'parquet'
    },
    'universe': {
        'path': DATA_PATH / "wrds_universe1.parquet",
        'extension': 'parquet'
    },
    'prices': {
        'path': DATA_PATH / "wrds_historical_prices1.parquet",
        'extension': 'parquet'
    }
}
RETURN_BOOL_UNIVERSE = False
SAVING_CONFIG_PRICES = {
    'prices': {
        'path': DATA_PATH / "wrds_universe_prices1.parquet",
        'extension': 'parquet'
    }
}

CONFIG1 = {
    "wrds_request":WRDS_REQUEST,
    "starting_date":STARTING_DATE,
    "ending_date":ENDING_DATE,
    "date_cols":DATE_COLS,
    "saving_config_universe":SAVING_CONFIG_UNIVERSE,
    "return_bool_universe":RETURN_BOOL_UNIVERSE,
    "saving_config_prices":SAVING_CONFIG_PRICES
}

STARTING_DATE2 = "2010-01-01" # of the wrds query YYYY-MM-DD
ENDING_DATE2 = None

SAVING_CONFIG_UNIVERSE2 = {
    'gross_query': {
        'path': DATA_PATH / "wrds_gross_query2.parquet",
        'extension': 'parquet'
    },
    'universe': {
        'path': DATA_PATH / "wrds_universe2.parquet",
        'extension': 'parquet'
    },
    'prices': {
        'path': DATA_PATH / "wrds_historical_prices2.parquet",
        'extension': 'parquet'
    }
}

SAVING_CONFIG_PRICES2 = {
    'prices': {
        'path': DATA_PATH / "wrds_universe_prices2.parquet",
        'extension': 'parquet'
    }
}

CONFIG2 = {
    "wrds_request":WRDS_REQUEST,
    "starting_date":STARTING_DATE2,
    "ending_date":ENDING_DATE2,
    "date_cols":DATE_COLS,
    "saving_config_universe":SAVING_CONFIG_UNIVERSE2,
    "return_bool_universe":RETURN_BOOL_UNIVERSE,
    "saving_config_prices":SAVING_CONFIG_PRICES2
}

CONFIGS = [CONFIG1, CONFIG2]