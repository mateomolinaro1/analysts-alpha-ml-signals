import polars as pl
import logging
from datetime import date

from .book_engine import BookEngine
from .features_engine import FeaturesEngine
from .model_wrappers.model_wrapper import ModelWrapper
from .meta_portfolio import MetaPortfolio
from .utils.config import Config
from .utils.mapping import ModelType
from .utils.helpers import Timeline, _parse_date, _add_months, _month_range
from .utils.s3_utils import s3Utils

logger = logging.getLogger(__name__)


class Backtester:
    """
    High level orchestrator dedicated to the backtesting process of the strategy based on machine learning models.

    It uses high level components to perform the following tasks:
        - BookEngine: Process the analysts' target prices to generate an implied portfolio.
        - FeatureEngine: Process analysts' books to generate features for the models.
        - ModelWrapper(s): Load trained machine learning models to perform inference on the generated features. 
        - MetaPortfolio: Construct and evaluate the strategy portfolio based on model predictions with a given methodology.

    Note: The models are assumed to be already trained and saved on disk by a separate component (Orchestrator).
          This part of training and saving is not handled here and is considered abstracted via the wrapper interface.
          Whatever model is used, it should be loadable and usable via the ModelWrapper interface.

    The backtesting is done as follows:
        0. Data is available from 2000 to 2025 - split into train (2000-2020) and test (2020-2025).
        1. A warm-up period is defined (e.g., 2018-2020) to allow feature generation prior to backtesting.
        2. For each month of the test period (2020-2025):
            a. Generate books from available analysts' target prices up to that month (x months of validity).
            b. Generate features from the generated books (current month and historical data).
            c. Load the trained model from disk.
            d. Perform inference on the generated features to obtain predictions.
            e. Construct the strategy portfolio based on model predictions using a defined methodology.
        3. Analyze the performance of the strategy over the backtest period using various metrics and visualizations.
    """

    def __init__(self, config: Config):
        """
        Parameters
        ----------
        config : Config
            Configuration object holding settings for the backtesting process.
        """
        self.config = config
        self.model: ModelWrapper = None
        self.ptf: MetaPortfolio = MetaPortfolio(dead_zone=self.config.transfo_dead_zone,
                                                k_pos=self.config.transfo_k_pos,
                                                k_neg=self.config.transfo_k_neg)
        self._load_model()

    def run(self):
        """
        Run the backtesting process as described in the class docstring.
        """
        logger.info("Starting backtesting process...")
        logger.info(f"Start Date: {self.config.test_start_date}, End Date: {self.config.test_end_date}")

        timeline = self._build_timeline()

        df_tp, df_prices = self._load_data(start_date=timeline.warmup[0], end_date=timeline.backtest[-1])

        book_history: list[pl.DataFrame] = []
        weight_history: list[pl.DataFrame] = []

        for t in timeline.warmup:
            logger.info(f"\tBuilding book for warm-up date: {t}")
            book_t = BookEngine(df_tp=df_tp,
                                df_prices=df_prices,
                                validity_length=self.config.validity_length,
                                decay_half_life=self.config.decay_halflife).at_snapshot(snapshot_date=t)
            book_history.append(book_t)

        for t in timeline.backtest:
            logger.info(f"\tRunning backtest for date: {t}")

            book_t = BookEngine(df_tp=df_tp,
                                df_prices=df_prices,
                                validity_length=self.config.validity_length,
                                decay_half_life=self.config.decay_halflife).at_snapshot(snapshot_date=t)
            book_history.append(book_t)

            X_t = FeaturesEngine(...).generate_features(book=book_t, as_of_date=t)
            y_hat_t = self.model.predict(X=X_t)

            w_t = self.ptf.create_metaportfolio(
                analyst_books=book_t,
                predict_pnl=y_hat_t,
                method=self.config.construction_method,
                normalize_weights=self.config.normalize_weights
            )
            weight_history.append(w_t)

        logger.info("Backtesting process completed.")

    # -----------------------------------------------------------------
    # |                       Private Helpers                         |
    # -----------------------------------------------------------------

    def _load_model(self) -> ModelWrapper:
        """
        Load the trained model from disk using the ModelWrapper interface.

        Returns
        -------
        ModelWrapper
            An instance of the wrapped model loaded from disk inherited from ModelWrapper.
        """
        self.model = ModelType[self.config.model_name].value()
        self.model.load_model(model_path=self.config.model_paths.get('SKOPS'),
                            manifest_path=self.config.model_paths.get('MANIFEST'))
        logger.info(f"Model {self.config.model_name} loaded successfully from disk.")
        
    def _load_data(self, start_date: date = None, end_date: date = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load data from S3 using paths defined in the configuration.

        Parameters
        ----------
        start_date : date, optional
            Start date to filter data for backtesting, by default None
        end_date : date, optional
            End date to filter data for backtesting, by default None

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the target prices DataFrame and prices DataFrame.
        """
        df_tp = s3Utils.pull_parquet_file_from_s3(path=self.config.target_price_path, to_polars=True)
        df_prices = s3Utils.pull_parquet_file_from_s3(path=self.config.prices_path, to_polars=True)

        if start_date and end_date:
            df_tp = df_tp.filter((pl.col("reco_date") >= start_date) & (pl.col("reco_date") <= end_date))
            df_prices = df_prices.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

        logger.info(f"Data loaded from S3: Target Prices ({df_tp.shape}), Prices ({df_prices.shape})")
        return df_tp, df_prices
    
    def _build_timeline(self) -> Timeline:
        """
        Build the timeline for the backtesting process, including warm-up and backtest periods.
        Each date in the timeline corresponds to the first day of each month.

        Example :

            Input:
                - test_start_date: str (e.g., "2020-01")
                - test_end_date: str (e.g., "2025-01")
                - warmup_test_months: int (e.g., 12)
            Output:
                - warmup: ["2019-01-01", ..., "2019-12-01"]
                - backtest: ["2020-01-01", ..., "2025-01-01"]
            
        Returns
        -------
        Timeline
            An object containing lists of dates for warm-up and backtest periods.
        """
        start_date = _parse_date(self.config.test_start_date)
        end_date = _parse_date(self.config.test_end_date)
        warmup_months = self.config.warmup_test_months

        if warmup_months > 0:
            warmup_start = _add_months(start_date, -warmup_months)
            warmup_end = _add_months(start_date, -1)
            warmup = _month_range(warmup_start, warmup_end)
        else:
            warmup = []

        backtest = _month_range(start_date, end_date)
        
        logger.info(f"Backtesting timeline constructed: Warm-up ({len(warmup)} months), Backtest ({len(backtest)} months)")
        return Timeline(warmup=warmup, backtest=backtest, all=warmup + backtest)
