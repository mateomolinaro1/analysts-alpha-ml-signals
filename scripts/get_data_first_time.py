import pandas as pd
from configs.config_get_data_first_time  import CONFIGS, LOG_PATH, DATA_PATH
from src.alpha_in_analysts.data.data_wrds import DataHandler
from src.alpha_in_analysts.utils.files_utils import FileUtils
from src.alpha_in_analysts.utils.s3_utils import s3Utils
from dotenv import load_dotenv
load_dotenv()
import os
import logging

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

FileUtils.delete_all_files(path=DATA_PATH, except_git_keep=True)

for CONFIG in CONFIGS:
    dh = DataHandler(
        data_path=DATA_PATH,
        wrds_username=os.getenv("WRDS_USERNAME"),
        wrds_password=os.getenv("WRDS_PASSWORD")
    )
    dh.connect_wrds()

    dh.fetch_wrds_historical_universe(wrds_request=CONFIG["wrds_request"],
                                      starting_date=CONFIG["starting_date"],
                                      ending_date=CONFIG["ending_date"],
                                      date_cols=CONFIG["date_cols"],
                                      saving_config=CONFIG["saving_config_universe"],
                                      return_bool=CONFIG["return_bool_universe"])
    dh.get_wrds_historical_prices(saving_config=CONFIG["saving_config_prices"])
    dh.get_wrds_returns()

    dh.logout_wrds()

logging.info("Data fetching and processing completed.")

# Need to concatenate and push to s3
to_load = [
    "wrds_gross_query1.parquet",
    "wrds_gross_query2.parquet",
    "wrds_universe1.parquet",
    "wrds_universe2.parquet",
    "wrds_universe_prices1.parquet",
    "wrds_universe_prices2.parquet",
]

# sanity check
if len(to_load) % 2 != 0:
    raise ValueError("to_load must contain an even number of files")

for i in range(0, len(to_load), 2):
    file_1 = to_load[i]
    file_2 = to_load[i + 1]

    print(f"Loading {file_1} and {file_2}")

    df1 = pd.read_parquet(DATA_PATH / file_1)
    df2 = pd.read_parquet(DATA_PATH / file_2)

    df_concat = pd.concat([df1, df2], axis=0, ignore_index=True)

    # output filename (remove trailing 1/2)
    base_name = file_1.rstrip("12.parquet") + ".parquet"
    output_path = DATA_PATH / base_name

    df_concat.to_parquet(output_path,
                         index=False,
                         engine="pyarrow",
                         compression="zstd"
                         )

    print(f"Saved → {output_path}")


