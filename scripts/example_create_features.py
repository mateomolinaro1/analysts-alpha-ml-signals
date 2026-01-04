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
from src.alpha_in_analysts.features_engine import FeaturesEngine

start = time.time()

feature_engine = FeaturesEngine(
    df_prices=df_prices,
    df_tp=df_tp,
    validity_length=12,
    decay_half_life=6,
    start_date="2000-01-31",
    end_date="2024-12-31",
)
feature_engine._build_pnl_all_analysts()
res = feature_engine.build_all_features(up_to_date="2024-12-31",
                                        lookback_perf_pct=12,
                                        lookback_perf=6,
                                        lookback_vol_pct=6,
                                        lookback_vol = 6,
                                        lookback_mean_ret=6,
                                        drop_na=True)