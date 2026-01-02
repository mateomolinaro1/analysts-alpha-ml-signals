import polars as pl
import time
from dotenv import load_dotenv
from src.alpha_in_analysts.utils.s3_utils import s3Utils

load_dotenv()
start = time.time()
df_tp = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/estimates.parquet", to_polars=True)
df_prices = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/prices.parquet", to_polars=True)

print("Data loaded from S3 in", round(time.time() - start, 2), "seconds")

from src.alpha_in_analysts.book_engine import BookEngine

start = time.time()
engine = BookEngine(df_tp=df_tp, df_prices=df_prices)
df_book = engine.at_snapshot(snapshot_date="2023-12-31")

print("Book computed at snapshot date in", round(time.time() - start, 2), "seconds")