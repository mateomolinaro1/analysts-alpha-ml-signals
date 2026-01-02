from src.alpha_in_analysts.utils.s3_utils import s3Utils
from src.alpha_in_analysts.data.data_wrds import DataHandler
from configs.config_get_data_first_time  import DATA_PATH
import os
import pandas as pd
import numpy as np
# import torch
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
load_dotenv()

# wrds_gq = s3Utils.pull_parquet_file_from_s3(
#     path="s3://alpha-in-analysts-storage/data/wrds_gross_query.parquet"
# )
# wrds_gq_monthly = (
#     wrds_gq.groupby(by=["date","permno"])
#     .resample('ME', on='date').last().reset_index()
# )

# Linking IBES to CRSP
dh = DataHandler(
    data_path=DATA_PATH,
    wrds_password=os.getenv("WRDS_PASSWORD"),
    wrds_username=os.getenv("WRDS_USERNAME")
)
dh.connect_wrds()
# 1.1 IBES: Get the list of IBES Tickers for US firms in IBES
_ibes1 = dh.wrds_db.raw_sql(
    """
    select ticker, cusip, cname, sdates
    from ibes.id
    where usfirm=1
        and cusip != ''
    """,
    date_cols=['sdates']
)
# Create first and last 'start dates' for a given cusip
# Use agg min and max to find the first and last date per group
# then rename to fdate and ldate respectively
_ibes1_date = (
    _ibes1
    .groupby(['ticker','cusip'])
    .sdates.agg(['min', 'max'])
    .reset_index()
    .rename(columns={'min':'fdate', 'max':'ldate'})
)
# merge fdate ldate back to _ibes1 data
_ibes2 = pd.merge(_ibes1, _ibes1_date,how='left', on =['ticker','cusip'])
_ibes2 = _ibes2.sort_values(by=['ticker','cusip','sdates'])
# keep only the most recent company name
# determined by having sdates = ldate
_ibes2 = _ibes2.loc[_ibes2.sdates == _ibes2.ldate].drop(['sdates'], axis=1)

# 1.2 CRSP: Get all permno-ncusip combinations
_crsp1 = dh.wrds_db.raw_sql(
    """
    select permno, ncusip, comnam, namedt, nameenddt
    from crsp.stocknames
    where ncusip != ''
    """,
    date_cols=['namedt', 'nameenddt']
)
# first namedt
_crsp1_fnamedt = (
    _crsp1
    .groupby(['permno','ncusip'])
    .namedt
    .min()
    .reset_index()
)

# last nameenddt
_crsp1_lnameenddt = (
    _crsp1
    .groupby(['permno','ncusip'])
    .nameenddt
    .max()
    .reset_index()
)

# merge both
_crsp1_dtrange = pd.merge(
    _crsp1_fnamedt,
    _crsp1_lnameenddt,
    on = ['permno','ncusip'],
    how='inner'
)

# replace namedt and nameenddt with the version from the dtrange
_crsp1 = (
    _crsp1
    .drop(['namedt'],axis=1)
    .rename(columns={'nameenddt':'enddt'})
)
_crsp2 = pd.merge(
    _crsp1,
    _crsp1_dtrange,
    on =['permno','ncusip'],
    how='inner'
)

# keep only most recent company name
_crsp2 = (
    _crsp2
    .loc[_crsp2.enddt ==_crsp2.nameenddt]
    .drop(['enddt'], axis=1)
)

# 1.3 Create CUSIP Link Table
# Link by full cusip, company names and dates
_link1_1 = (
    pd.merge(_ibes2, _crsp2, how='inner', left_on='cusip', right_on='ncusip')
    .sort_values(['ticker','permno','ldate'])
)
# Keep link with most recent company name
_link1_1_tmp = _link1_1.groupby(['ticker','permno']).ldate.max().reset_index()
_link1_2 = pd.merge(_link1_1, _link1_1_tmp, how='inner', on =['ticker', 'permno', 'ldate'])
_link1_2['name_ratio'] = _link1_2.apply(lambda x: fuzz.token_set_ratio(x.comnam, x.cname), axis=1)
name_ratio_p10 = _link1_2.name_ratio.quantile(0.10)

def score1(row):
    if (row['fdate']<=row['nameenddt']) & (row['ldate']>=row['namedt']) & (row['name_ratio'] >= name_ratio_p10):
        score = 0
    elif (row['fdate']<=row['nameenddt']) & (row['ldate']>=row['namedt']):
        score = 1
    elif row['name_ratio'] >= name_ratio_p10:
        score = 2
    else:
        score = 3
    return score

# assign size portfolio
_link1_2['score']=_link1_2.apply(score1, axis=1)
_link1_2 = _link1_2[['ticker','permno','cname','comnam','name_ratio','score']]
_link1_2 = _link1_2.drop_duplicates()

##########################
# Step 2: Link by TICKER #
##########################

# Find links for the remaining unmatched cases using Exchange Ticker

# Identify remaining unmatched cases
_nomatch1 = pd.merge(_ibes2[['ticker']], _link1_2[['permno','ticker']], on='ticker', how='left')
_nomatch1 = _nomatch1.loc[_nomatch1.permno.isnull()].drop(['permno'], axis=1).drop_duplicates()
# Add IBES identifying information

ibesid = dh.wrds_db.raw_sql(
    """
    select ticker, cname, oftic, sdates, cusip
    from ibes.id
    """,
    date_cols=['sdates']
)
ibesid = ibesid.loc[ibesid.oftic.notna()]
_nomatch2 = pd.merge(_nomatch1, ibesid, how='inner', on=['ticker'])
_nomatch3 = (
    _nomatch2
    .groupby(['ticker', 'oftic'])
    .sdates.agg(['min', 'max'])
    .reset_index()
    .rename(columns={'min':'fdate', 'max':'ldate'})
)
_nomatch3 = pd.merge(_nomatch2, _nomatch3, how='left', on=['ticker','oftic'])
_nomatch3 = _nomatch3.loc[_nomatch3.sdates == _nomatch3.ldate]

# Get entire list of CRSP stocks with Exchange Ticker information
_crsp_n1 = dh.wrds_db.raw_sql(
    """
    select ticker, comnam, permno, ncusip, namedt, nameenddt
    from crsp.stocknames
    """,
    date_cols=['namedt', 'nameenddt']
)
_crsp_n1 = _crsp_n1.loc[_crsp_n1.ticker.notna()].sort_values(by=['permno','ticker','namedt'])
# Arrange effective dates for link by Exchange Ticker

_crsp_n1_namedt = _crsp_n1.groupby(['permno','ticker']).namedt.min().reset_index().rename(columns={'min':'namedt'})
_crsp_n1_nameenddt = _crsp_n1.groupby(['permno','ticker']).nameenddt.max().reset_index().rename(columns={'max':'nameenddt'})

_crsp_n1_dt = pd.merge(_crsp_n1_namedt, _crsp_n1_nameenddt, how = 'inner', on=['permno','ticker'])

_crsp_n1 = _crsp_n1.rename(columns={'namedt': 'namedt_ind', 'nameenddt':'nameenddt_ind'})
_crsp_n2 = pd.merge(_crsp_n1, _crsp_n1_dt, how ='left', on = ['permno','ticker'])

_crsp_n2 = _crsp_n2.rename(columns={'ticker':'crsp_ticker'})
_crsp_n2 = _crsp_n2.loc[_crsp_n2.nameenddt_ind == _crsp_n2.nameenddt].drop(['namedt_ind', 'nameenddt_ind'], axis=1)
# Merge remaining unmatched cases using Exchange Ticker
# Note: Use ticker date ranges as exchange tickers are reused overtime

_link2_1 = pd.merge(_nomatch3, _crsp_n2, how='inner', left_on=['oftic'], right_on=['crsp_ticker'])
_link2_1 = _link2_1.loc[(_link2_1.ldate>=_link2_1.namedt) & (_link2_1.fdate<=_link2_1.nameenddt)]

# Score using company name using 6-digit CUSIP and company name spelling distance
_link2_1['name_ratio'] = _link2_1.apply(lambda x: fuzz.token_set_ratio(x.comnam, x.cname), axis=1)

_link2_2 = _link2_1
_link2_2['cusip6'] = _link2_2.apply(lambda x: x.cusip[:6], axis=1)
_link2_2['ncusip6'] = _link2_2.apply(lambda x: x.ncusip[:6], axis=1)

# Score using company name using 6-digit CUSIP and company name spelling distance

def score2(row):
    if (row['cusip6']==row['ncusip6']) & (row['name_ratio'] >= name_ratio_p10):
        score = 0
    elif (row['cusip6']==row['ncusip6']):
        score = 4
    elif row['name_ratio'] >= name_ratio_p10:
        score = 5
    else:
        score = 6
    return score

# assign size portfolio
_link2_2['score']=_link2_2.apply(score2, axis=1)
# Some companies may have more than one TICKER-PERMNO link
# so re-sort and keep the case (PERMNO & Company name from CRSP)
# that gives the lowest score for each IBES TICKER

_link2_2 = _link2_2[['ticker','permno','cname','comnam', 'name_ratio', 'score']].sort_values(by=['ticker','score'])
_link2_2_score = _link2_2.groupby(['ticker']).score.min().reset_index()

_link2_3 = pd.merge(_link2_2, _link2_2_score, how='inner', on=['ticker', 'score'])
_link2_3 = _link2_3[['ticker','permno','cname','comnam','score']].drop_duplicates()

#####################################
# Step 3: Finalize LInks and Scores #
#####################################
# Error
# iclink = _link1_2.append(_link2_3)

s3Utils.push_object_to_s3_parquet(
    object_to_push=_link1_2,
    path="s3://alpha-in-analysts-storage/data/ibes_crsp_linking_table.parquet"
)
e = s3Utils.pull_parquet_file_from_s3(
    path="s3://alpha-in-analysts-storage/data/estimates.parquet"
)

est = dh.wrds_db.raw_sql(
    """
    SELECT * 
    FROM ibes.ptgdet
    """
)
s3Utils.push_object_to_s3_parquet(
    object_to_push=est,
    path="s3://alpha-in-analysts-storage/data/target_prices_gross_query.parquet"
)


# Get crsp prices monthly
# At least, the query must retrieve the following columns:
# ['ticker','exchcd','cusip','ncusip','comnam','permno','permco','namedt','nameendt','date']
STARTING_DATE = "1999-01-01" # of the wrds query YYYY-MM-DD
ENDING_DATE = None
WRDS_REQUEST = """
SELECT
    a.ticker, a.exchcd,
    a.comnam, a.cusip, a.ncusip,
    a.permno, a.permco,
    a.namedt, a.nameendt,
    b.date, b.ret, b.prc, b.shrout, b.vol,
    ABS(b.prc) * b.shrout * 1000 AS market_cap
FROM crsp.msenames AS a
JOIN crsp.msf AS b
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
        'path': DATA_PATH / "wrds_gross_query_monthly.parquet",
        'extension': 'parquet'
    },
    'universe': {
        'path': DATA_PATH / "wrds_universe_monthly.parquet",
        'extension': 'parquet'
    },
    'prices': {
        'path': DATA_PATH / "wrds_historical_prices_monthly.parquet",
        'extension': 'parquet'
    }
}
RETURN_BOOL_UNIVERSE = False
SAVING_CONFIG_PRICES = {
    'prices': {
        'path': DATA_PATH / "wrds_universe_prices_monthly.parquet",
        'extension': 'parquet'
    }
}
dh.fetch_wrds_historical_universe(
    wrds_request=WRDS_REQUEST,
    starting_date=STARTING_DATE,
    ending_date=ENDING_DATE,
    date_cols=DATE_COLS,
    saving_config=SAVING_CONFIG_UNIVERSE,
    return_bool=RETURN_BOOL_UNIVERSE
)
s3Utils.push_object_to_s3_parquet(
    object_to_push=dh.wrds_gross_query,
    path="s3://alpha-in-analysts-storage/data/wrds_gross_query_monthly.parquet"
)

wrds_gq_monthly = s3Utils.pull_parquet_file_from_s3(
    path="s3://alpha-in-analysts-storage/data/wrds_gross_query_monthly.parquet"
)
ibes_crsp_linking_table = s3Utils.pull_parquet_file_from_s3(
    path="s3://alpha-in-analysts-storage/data/ibes_crsp_linking_table.parquet"
)
target_prices_gross_query = s3Utils.pull_parquet_file_from_s3(
    path="s3://alpha-in-analysts-storage/data/target_prices_gross_query.parquet"
)
wrds_gq_monthly.rename(columns={'ticker':'crsp_ticker'}, inplace=True)
ibes_crsp_linking_table.rename(columns={'ticker':'ibes_ticker'}, inplace=True)
target_prices_gross_query.rename(columns={'ticker':'ibes_ticker'}, inplace=True)
wrds_gq_monthly.drop(columns=['comnam'], inplace=True)
wrds_gq_monthly.rename(columns={'cusip':'crsp_cusip'}, inplace=True)
target_prices_gross_query.rename(columns={'cusip':'ibes_cusip'}, inplace=True)

merged_df = pd.merge(
    wrds_gq_monthly,
    ibes_crsp_linking_table,
    left_on=['permno'],
    right_on=['permno'],
    how='left'
)

# 1. Ensure datetime
merged_df['date'] = pd.to_datetime(merged_df['date'])
target_prices_gross_query['anndats'] = pd.to_datetime(target_prices_gross_query['anndats'])

# 2. Drop NaNs FIRST
merged_df = merged_df.dropna(subset=['ibes_ticker', 'date'])
target_prices_gross_query = target_prices_gross_query.dropna(subset=['ibes_ticker', 'anndats'])

# 3. Sort + reset index AFTER cleaning
merged_df = (
    merged_df
    .sort_values('date')
    .reset_index(drop=True)
)

target_prices_gross_query = (
    target_prices_gross_query
    .sort_values('anndats')
    .reset_index(drop=True)
)

merged_df = pd.merge_asof(
    merged_df,
    target_prices_gross_query,
    left_on='date',
    right_on='anndats',
    by = 'ibes_ticker',
    direction='backward',
    tolerance=pd.Timedelta('180D')
)
# delete the rows where horizon is not 12 months
merged_df['horizon'] = merged_df['horizon'].astype(float)
merged_df = merged_df.loc[merged_df['horizon'] == 12,:]

analysts = target_prices_gross_query["alysnam"].dropna().unique().tolist()
analysts = sorted(analysts)
tickers = merged_df["ibes_ticker"].dropna().unique().tolist()
tickers = sorted(tickers)
dates = merged_df["date"].dropna().unique().tolist()
dates = sorted(dates)

del target_prices_gross_query
del wrds_gq_monthly

# date_to_d = {d: i for i, d in enumerate(dates)}
# ticker_to_k = {t: j for j, t in enumerate(tickers)}
# analyst_to_i = {a: k for k, a in enumerate(analysts)}
#
# d = merged_df['date'].map(date_to_d).to_numpy()
# k = merged_df['ibes_ticker'].map(ticker_to_k).to_numpy()
# i = merged_df['alysnam'].map(analyst_to_i).to_numpy()
#
# mask = ~np.isnan(i)
# d, k, i = d[mask], k[mask], i[mask]
#
# T = len(dates)
# I = len(analysts)
# K = len(tickers)
#
# X = torch.full((T, I, K), float('nan'))
# values = torch.tensor(
#     merged_df.loc[mask, 'value'].to_numpy(),
#     dtype=torch.float32
# )
# X[d, i, k] = values
tp = {}
for d in dates:
    df_at_d = merged_df.loc[merged_df['date'] == d]
    # Pre define empty DataFrame
    tp[d] = pd.DataFrame(
        data = np.nan,
        index = df_at_d['alysnam'].unique(),
        columns = df_at_d['ibes_ticker'].unique()
    )
    for _, row in df_at_d.iterrows():
        tp[d].at[row['alysnam'], row['ibes_ticker']] = row['value']

mkt_prices_df = merged_df.pivot_table(
    index='date',
    columns='ibes_ticker',
    values='prc',
    aggfunc='first'
)
n = 12
mkt_prices_df_forward = mkt_prices_df.shift(-n)
dates_fwd = mkt_prices_df.index[~mkt_prices_df.shift(n).isna().all(axis=1)]
merged_df_fwd = merged_df.groupby([''])

mkt_prices = {}
keys_tp = list(tp.keys())
for idx,d in enumerate(dates):
    df_at_d = mkt_prices_df_forward.loc[d]
    # Pre define empty DataFrame
    mkt_prices[d] = pd.DataFrame(
        data = np.nan,
        index = [d],
        columns = tp[keys_tp[idx]].columns
    )

    for ibes_ticker in df_at_d.index:
        if ibes_ticker in mkt_prices[d].columns:
            mkt_prices[d].loc[d, ibes_ticker] = df_at_d.loc[ibes_ticker]
        else:
            continue

upside = {}
mapping_dates_fwd_dates = {d:fwd_date for d,fwd_date in zip(dates, dates_fwd)}
for k,v in mapping_dates_fwd_dates.items():
    upside[v] = (tp[k].values - mkt_prices[k].values) / mkt_prices[k].values





