import polars as pl
from typing import Union

class CreatePortfolio:
    """Class to compute portfolio level returns given assets' weights and returns"""

    def __init__(self,
                 weights: pl.DataFrame,
                 returns: pl.DataFrame,
                 rebal_periods: int = 0
                 ):
        self.weights = weights
        self.returns = returns
        self.rebal_periods = rebal_periods

        self.turnover = None
        self.rebalanced_weights = None


    def rebalance_portfolio(
            self,
            return_bool: bool = False
    ) -> Union[None, pl.DataFrame]:
        """
        Rebalance the portfolio and compute weights evolution between rebalancing dates.
        Returns rebalanced weights.

        Parameters:
        - return_bool: return the df is set to true otherwise just stored

        Returns:
        - None or df: rebalanced_weights
        """
        # If no rebalancing period specified or rebalancing at every period, return original weights
        if self.rebal_periods is None or self.rebal_periods == 0:
            if return_bool:
                return self.weights

        if not self.weights.schema["date"] in (pl.Date, pl.Datetime):
            raise ValueError("weights dataframe must have a 'date' column of type Date or Datetime")

        # If rebalancing period specified, we must compute weights accounting for drift
        # Step 0 Initialize the rebalanced_weights df
        self.rebalanced_weights = self.weights.clone()
        self.turnover = self.weights.select(
            "date",
            pl.lit(None).cast(pl.Float64).alias("turnover")
        )

        # Step 1 define the rebalancing dates range
        weights_only = pl.all().exclude("date")
        dates = (
            self.weights
            .with_columns(
                pl.sum_horizontal(weights_only).alias("row_sum")
            )
            .filter(pl.col("row_sum").is_not_null())
            .select("date")
            .to_series()
            .to_list()
        )

        start_date = dates[0]
        rebal_dates = dates[::self.rebal_periods]
        rebal_dates.pop(0)

        # Step 2: loop on all the dates
        for date in self.rebalanced_weights.select("date").to_series().to_list():
            if date < start_date:
                # we do nothing as it is already set to nan
                continue
            elif ((date > start_date) and (date in rebal_dates)) or date == start_date:
                # we do nothing because being at a rebalancing date means that we "reset" the weights
                # to EW and self.computes weights does that to every dates by default

                # Compute turnover
                num_idx = (
                    self.rebalanced_weights
                    .with_row_index()
                    .filter(pl.col("date") == date)
                    .select("index")
                    .item()
                )

                weights_only = pl.all().exclude("date")

                curr = (
                    self.rebalanced_weights
                    .slice(num_idx, 1)
                    .select(weights_only)
                )

                prev = (
                    self.rebalanced_weights
                    .slice(num_idx - 1, 1)
                    .select(weights_only)
                )

                turnover = (
                    (curr.fill_null(0.0) - prev.fill_null(0.0))
                    .select(pl.sum_horizontal(pl.all().abs()))
                    .item()
                )

                self.turnover = self.turnover.with_columns(
                    pl.when(pl.col("date") == date)
                    .then(turnover)
                    .otherwise(pl.col("turnover"))
                    .alias("turnover")
                )


            elif (date > start_date) and (date not in rebal_dates):
                # If we are not at a rebalancing date, we must derive the weights in between rebal dates
                weights_cols = pl.all().exclude("date")

                num_idx = (
                    self.rebalanced_weights
                    .with_row_index()
                    .filter(pl.col("date") == date)
                    .select("index")
                    .item()
                )

                prev_weights = (
                    self.rebalanced_weights
                    .slice(num_idx - 1, 1)
                    .select(weights_cols)
                )

                aligned_ret = (
                    self.returns
                    .slice(num_idx, 1)
                    .select(weights_cols)
                )

                drifted_w = (
                        prev_weights
                        * (1 + aligned_ret)
                )

                drifted_w = drifted_w / drifted_w.sum_horizontal()

                # write back
                self.rebalanced_weights = self.rebalanced_weights.with_columns([
                    pl.when(pl.col("date") == date)
                    .then(drifted_w[col])
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in self.rebalanced_weights.columns
                    if col != "date"
                ])

            else:
                continue