import pandas as pd
import numpy as np
from datetime import date
from typing import List, Union, Tuple, Dict, Optional
import logging
import wrds
from pathlib import Path

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self,
                 data_path: Path,
                 wrds_username:str,
                 wrds_password:str,
    )->None:

        self.data_path = data_path
        self.wrds_username = wrds_username
        self.wrds_password = wrds_password

        self.wrds_db = None
        self.wrds_gross_query = None
        self.wrds_universe = None
        self.wrds_universe_last_date = None
        self.universe_prices_wrds = None
        self.universe_returns_wrds = None
        self.fields_wrds_to_keep_for_universe = ['ticker',
                                                 'exchcd',
                                                 'cusip',
                                                 'ncusip',
                                                 'comnam',
                                                 'permno',
                                                 'permco',
                                                 'namedt',
                                                 'nameendt',
                                                 'date']


    def connect_wrds(self):
        """Establishes a connection to the WRDS database using the provided username."""
        self.wrds_db = wrds.Connection(wrds_username=self.wrds_username,
                                       wrds_password=self.wrds_password)

    def logout_wrds(self):
        """Logs out from the WRDS database connection."""
        if self.wrds_db is not None:
            self.wrds_db.close()
            self.wrds_db = None

    # def load_data(self):
    #     # Load WRDS universe
    #     if self.wrds_universe is None:
    #         try:
    #             self.wrds_universe = pd.read_parquet(self.data_path / "wrds_universe.parquet")
    #         except Exception as e:
    #             logger.error(f"Error reading WRDS universe: {e}")
    #             raise ValueError("wrds_universe data is not loaded. Please fetch it first.")
    #
    #     return

    def fetch_wrds_historical_universe(self,
                                       wrds_request:str,
                                       starting_date:str,
                                       ending_date:str|None,
                                       date_cols:List[str],
                                       saving_config:dict,
                                       return_bool:bool=False,
                                       )->Union[None,dict]:
        """
        Fetches historical universe from WRDS based on the provided SQL request. It saves wrds_gross_query
        and wrds_universe to disk if specified in saving_config.
        :parameters:
        - wrds_request: SQL query string to fetch data from WRDS.
        - date_cols: List of columns in the query result that should be parsed as dates.
        - saving_config: Dictionary specifying saving paths and formats for gross query and universe.
        - return_bool: If True, returns the fetched universe DataFrame.
        """
        # Check input data types
        if not isinstance(wrds_request, str):
            logger.error("wrds_request must be a string.")
            raise ValueError("wrds_request must be a string containing the SQL query.")
        if not isinstance(starting_date, str):
            logger.error("starting_date must be a string.")
            raise ValueError("starting_date must be a string in 'YYYY-MM-DD' format.")
        if ending_date is not None and not isinstance(ending_date, str):
            logger.error("ending_date must be a string or None.")
            raise ValueError("starting_date must be a string in 'YYYY-MM-DD' format.")
        if not isinstance(date_cols, list):
            logger.error("date_cols must be a list of strings.")
            raise ValueError("date_cols must be a list of strings.")
        for col in date_cols:
            if not isinstance(col, str):
                logger.error("All elements in date_cols must be strings.")
                raise ValueError("All elements in date_cols must be strings.")
        if not isinstance(saving_config, dict):
            logger.error("saving_config must be a dictionary.")
            raise ValueError("saving_config must be a dictionary.")
        if not isinstance(return_bool, bool):
            logger.error("return_bool must be a boolean.")
            raise ValueError("return_bool must be a boolean.")

        # Ensure connection to WRDS
        if self.wrds_db is None:
            self.connect_wrds()

        if ending_date is None:
            ending_date = date.today().strftime("%Y-%m-%d")

        # Query WRDS database
        wrds_request = wrds_request.format(starting_date=starting_date,
                                           ending_date=ending_date)

        self.wrds_gross_query = self.wrds_db.raw_sql(sql=wrds_request,
                                                     date_cols=date_cols)


        # unique identifiers of WRDS/CRSP are PERMNO
        self.wrds_gross_query = self.wrds_gross_query.drop_duplicates(subset=['date', 'permno'],
                                                                      keep='last')
        # Sort for checking
        self.wrds_gross_query = self.wrds_gross_query.sort_values(by=['date'],
                                                                  ascending=True).reset_index(drop=True)

        # Save gross query if specified
        if 'gross_query' in saving_config:
            if saving_config['gross_query']['extension'] == 'parquet':
                self.wrds_gross_query.to_parquet(saving_config['gross_query']['path'],
                                                index=False)
            else:
                logger.error("Unsupported file extension for gross query.")
                raise ValueError("Unsupported file extension for gross query. Use 'parquet'.")


        universe = self.wrds_gross_query.copy()
        universe.index = universe['date']
        self.wrds_universe = universe

        # Save to file if a saving path is provided
        if 'universe' in saving_config:
            if saving_config['universe']['extension'] == 'parquet':
                self.wrds_universe.to_parquet(saving_config['universe']['path'],
                                              index=True)
            else:
                logger.error("Unsupported file extension for universe.")
                raise ValueError("Unsupported file extension for universe. Use 'parquet'.")

        if return_bool:
            return {'wrds_gross_query':self.wrds_gross_query,
                    'wrds_universe':self.wrds_universe}

    def get_wrds_historical_prices(self,
                                   saving_config:dict,
                                   return_bool:bool=False) -> Union[None, pd.DataFrame]:
        """
        Format self.wrds_gross_query to have a nice prices df.
        :parameters:
        - saving_config: Dictionary specifying saving paths and formats for prices.
        - return_bool: If True, returns the prices DataFrame.
        It either saves the prices DataFrame to disk or returns it based on the parameters.
        """
        if self.wrds_gross_query is None:
            logger.error(f"Error reading WRDS gross query")
            raise ValueError("WRDS universe data is not loaded. Please fetch it first.")

        prices = self.wrds_gross_query.pivot(values='prc',
                                             index='date',
                                             columns='permno')
        self.universe_prices_wrds = prices

        if 'prices' in saving_config:
            if saving_config['prices']['extension'] == 'parquet':
                prices.to_parquet(saving_config['prices']['path'],
                              index=True)
            else:
                raise ValueError("Unsupported file extension for prices. Use 'parquet'.")

        if return_bool:
            return prices

    def get_wrds_returns(self,
                         return_bool:bool=False) -> Union[None, pd.DataFrame]:
        """
        Compute returns DataFrame from universe prices
        :parameters:
        - return_bool: If True, returns the returns DataFrame.
        """
        if self.universe_prices_wrds is None:
            raise ValueError("Universe prices data is not loaded. Please fetch it first.")
        returns = self.universe_prices_wrds.pct_change(fill_method=None)
        self.universe_returns_wrds = returns
        if return_bool:
            return returns

    def get_ibes_crsp_linking_table(self):
        # 1.1 IBES: Get the list of IBES Tickers for US firms in IBES
        _ibes1 = self.wrds_db.raw_sql(
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
            .groupby(['ticker', 'cusip'])
            .sdates.agg(['min', 'max'])
            .reset_index()
            .rename(columns={'min': 'fdate', 'max': 'ldate'})
        )
        # merge fdate ldate back to _ibes1 data
        _ibes2 = pd.merge(_ibes1, _ibes1_date, how='left', on=['ticker', 'cusip'])
        _ibes2 = _ibes2.sort_values(by=['ticker', 'cusip', 'sdates'])
        # keep only the most recent company name
        # determined by having sdates = ldate
        _ibes2 = _ibes2.loc[_ibes2.sdates == _ibes2.ldate].drop(['sdates'], axis=1)

        # 1.2 CRSP: Get all permno-ncusip combinations
        _crsp1 = self.wrds_db.raw_sql(
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
            .groupby(['permno', 'ncusip'])
            .namedt
            .min()
            .reset_index()
        )

        # last nameenddt
        _crsp1_lnameenddt = (
            _crsp1
            .groupby(['permno', 'ncusip'])
            .nameenddt
            .max()
            .reset_index()
        )

        # merge both
        _crsp1_dtrange = pd.merge(
            _crsp1_fnamedt,
            _crsp1_lnameenddt,
            on=['permno', 'ncusip'],
            how='inner'
        )

        # replace namedt and nameenddt with the version from the dtrange
        _crsp1 = (
            _crsp1
            .drop(['namedt'], axis=1)
            .rename(columns={'nameenddt': 'enddt'})
        )
        _crsp2 = pd.merge(
            _crsp1,
            _crsp1_dtrange,
            on=['permno', 'ncusip'],
            how='inner'
        )

        # keep only most recent company name
        _crsp2 = (
            _crsp2
            .loc[_crsp2.enddt == _crsp2.nameenddt]
            .drop(['enddt'], axis=1)
        )

        # 1.3 Create CUSIP Link Table
        # Link by full cusip, company names and dates
        _link1_1 = (
            pd.merge(_ibes2, _crsp2, how='inner', left_on='cusip', right_on='ncusip')
            .sort_values(['ticker', 'permno', 'ldate'])
        )
        # Keep link with most recent company name
        _link1_1_tmp = _link1_1.groupby(['ticker', 'permno']).ldate.max().reset_index()
        _link1_2 = pd.merge(_link1_1, _link1_1_tmp, how='inner', on=['ticker', 'permno', 'ldate'])
        _link1_2['name_ratio'] = _link1_2.apply(lambda x: fuzz.token_set_ratio(x.comnam, x.cname), axis=1)
        name_ratio_p10 = _link1_2.name_ratio.quantile(0.10)

        def score1(row):
            if (row['fdate'] <= row['nameenddt']) & (row['ldate'] >= row['namedt']) & (
                    row['name_ratio'] >= name_ratio_p10):
                score = 0
            elif (row['fdate'] <= row['nameenddt']) & (row['ldate'] >= row['namedt']):
                score = 1
            elif row['name_ratio'] >= name_ratio_p10:
                score = 2
            else:
                score = 3
            return score

        # assign size portfolio
        _link1_2['score'] = _link1_2.apply(score1, axis=1)
        _link1_2 = _link1_2[['ticker', 'permno', 'cname', 'comnam', 'name_ratio', 'score']]
        _link1_2 = _link1_2.drop_duplicates()

        ##########################
        # Step 2: Link by TICKER #
        ##########################

        # Find links for the remaining unmatched cases using Exchange Ticker

        # Identify remaining unmatched cases
        _nomatch1 = pd.merge(_ibes2[['ticker']], _link1_2[['permno', 'ticker']], on='ticker', how='left')
        _nomatch1 = _nomatch1.loc[_nomatch1.permno.isnull()].drop(['permno'], axis=1).drop_duplicates()
        # Add IBES identifying information

        ibesid = self.wrds_db.raw_sql(
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
            .rename(columns={'min': 'fdate', 'max': 'ldate'})
        )
        _nomatch3 = pd.merge(_nomatch2, _nomatch3, how='left', on=['ticker', 'oftic'])
        _nomatch3 = _nomatch3.loc[_nomatch3.sdates == _nomatch3.ldate]

        # Get entire list of CRSP stocks with Exchange Ticker information
        _crsp_n1 = self.wrds_db.raw_sql(
            """
            select ticker, comnam, permno, ncusip, namedt, nameenddt
            from crsp.stocknames
            """,
            date_cols=['namedt', 'nameenddt']
        )
        _crsp_n1 = _crsp_n1.loc[_crsp_n1.ticker.notna()].sort_values(by=['permno', 'ticker', 'namedt'])
        # Arrange effective dates for link by Exchange Ticker

        _crsp_n1_namedt = _crsp_n1.groupby(['permno', 'ticker']).namedt.min().reset_index().rename(
            columns={'min': 'namedt'})
        _crsp_n1_nameenddt = _crsp_n1.groupby(['permno', 'ticker']).nameenddt.max().reset_index().rename(
            columns={'max': 'nameenddt'})

        _crsp_n1_dt = pd.merge(_crsp_n1_namedt, _crsp_n1_nameenddt, how='inner', on=['permno', 'ticker'])

        _crsp_n1 = _crsp_n1.rename(columns={'namedt': 'namedt_ind', 'nameenddt': 'nameenddt_ind'})
        _crsp_n2 = pd.merge(_crsp_n1, _crsp_n1_dt, how='left', on=['permno', 'ticker'])

        _crsp_n2 = _crsp_n2.rename(columns={'ticker': 'crsp_ticker'})
        _crsp_n2 = _crsp_n2.loc[_crsp_n2.nameenddt_ind == _crsp_n2.nameenddt].drop(['namedt_ind', 'nameenddt_ind'],
                                                                                   axis=1)
        # Merge remaining unmatched cases using Exchange Ticker
        # Note: Use ticker date ranges as exchange tickers are reused overtime

        _link2_1 = pd.merge(_nomatch3, _crsp_n2, how='inner', left_on=['oftic'], right_on=['crsp_ticker'])
        _link2_1 = _link2_1.loc[(_link2_1.ldate >= _link2_1.namedt) & (_link2_1.fdate <= _link2_1.nameenddt)]

        # Score using company name using 6-digit CUSIP and company name spelling distance
        _link2_1['name_ratio'] = _link2_1.apply(lambda x: fuzz.token_set_ratio(x.comnam, x.cname), axis=1)

        _link2_2 = _link2_1
        _link2_2['cusip6'] = _link2_2.apply(lambda x: x.cusip[:6], axis=1)
        _link2_2['ncusip6'] = _link2_2.apply(lambda x: x.ncusip[:6], axis=1)

        # Score using company name using 6-digit CUSIP and company name spelling distance

        def score2(row):
            if (row['cusip6'] == row['ncusip6']) & (row['name_ratio'] >= name_ratio_p10):
                score = 0
            elif (row['cusip6'] == row['ncusip6']):
                score = 4
            elif row['name_ratio'] >= name_ratio_p10:
                score = 5
            else:
                score = 6
            return score

        # assign size portfolio
        _link2_2['score'] = _link2_2.apply(score2, axis=1)
        # Some companies may have more than one TICKER-PERMNO link
        # so re-sort and keep the case (PERMNO & Company name from CRSP)
        # that gives the lowest score for each IBES TICKER

        _link2_2 = _link2_2[['ticker', 'permno', 'cname', 'comnam', 'name_ratio', 'score']].sort_values(
            by=['ticker', 'score'])
        _link2_2_score = _link2_2.groupby(['ticker']).score.min().reset_index()

        _link2_3 = pd.merge(_link2_2, _link2_2_score, how='inner', on=['ticker', 'score'])
        _link2_3 = _link2_3[['ticker', 'permno', 'cname', 'comnam', 'score']].drop_duplicates()

        return _link1_2


