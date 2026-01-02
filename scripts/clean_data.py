import polars as pl

# Load gross data from S3
from dotenv import load_dotenv
from src.alpha_in_analysts.utils.s3_utils import s3Utils

load_dotenv()
wrds_gq_monthly = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/wrds_gross_query_monthly.parquet")
ibes_crsp_linking_table = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/ibes_crsp_linking_table.parquet")
target_prices_gross_query = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/target_prices_gross_query.parquet")

# Cleaning
df_tp = pl.DataFrame(target_prices_gross_query)
df_prices = pl.DataFrame(wrds_gq_monthly)
df_linking = pl.DataFrame(ibes_crsp_linking_table)

df_tp = (
    df_tp
    .select([
        pl.col("anndats").alias("reco_date").cast(pl.Date),
        pl.col("amaskcd").alias("analyst_id").cast(pl.Int64),
        pl.col("ticker"),
        pl.col("value").alias("target_price").abs().cast(pl.Float64),
        pl.col("curr").alias("currency"),
    ])
    .filter(pl.col("reco_date") >= pl.datetime(2000, 1, 1))
    .drop_nulls()
)

df_prices = (
    df_prices
    .select([
        pl.col("date").cast(pl.Date),
        pl.col("prc").abs().alias("price").cast(pl.Float64),
        pl.col("permno").alias("stock_id")
    ])
    .filter(pl.col("date") >= pl.datetime(2000, 1, 1))
    .drop_nulls()
)

from collections import Counter
ccy_counts = Counter(df_tp["currency"].to_list())
print("Currency counts:", ccy_counts)

df_tp = df_tp.filter(pl.col("currency") == "USD").drop("currency")

df_tp =(
    df_tp
    .join(
        df_linking.select([pl.col("ticker"), pl.col("permno")]),
        on="ticker",
        how="left"
    )
    .drop("ticker")
    .rename({"permno": "stock_id"})
)
df_tp = df_tp.drop_nulls(subset=["stock_id", "reco_date", "target_price", "analyst_id"])

valid_ids = df_tp["stock_id"].unique().to_list()
df_prices = df_prices.filter(pl.col("stock_id").is_in(valid_ids))

# Push cleaned data to S3
s3Utils.push_object_to_s3_parquet(
    object_to_push=df_tp.to_pandas(),
    path="s3://alpha-in-analysts-storage/data/estimates.parquet"
)
s3Utils.push_object_to_s3_parquet(
    object_to_push=df_prices.to_pandas(),
    path="s3://alpha-in-analysts-storage/data/prices.parquet"
)
print("Done")