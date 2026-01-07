import pandas as pd
import polars as pl
import time
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from src.alpha_in_analysts.utils.s3_utils import s3Utils
from src.alpha_in_analysts.features_engine import FeaturesEngine
from src.alpha_in_analysts.utils.config import Config

#-----------------------------------------------------------------------------
# Create features
#-----------------------------------------------------------------------------
config = Config()
start = time.time()
df_tp = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/estimates.parquet", to_polars=True)
df_prices = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/prices.parquet", to_polars=True)
print("Data loaded from S3 in", round(time.time() - start, 2), "seconds")

feature_engine = FeaturesEngine(
    config=config,
    df_prices=df_prices,
    df_tp=df_tp
)
# feature_engine._build_y()
# feature_engine._build_pnl_all_analysts()
# feature_engine.pnl_all_analysts.write_parquet("pnl_all_analysts_aligned_ret_ffill2m.parquet")
try:
    all_features = s3Utils.pull_parquet_file_from_s3(
        path="s3://alpha-in-analysts-storage/data/all_features.parquet"
    )
except Exception as e:
    print("all_features not on s3, computing it.")
    for i, date in enumerate(feature_engine.dates[24:]):  # skip first 24 months to have enough lookback
        print("Building features at date:", date)
        res = feature_engine.get_features_and_y(
            up_to_date=str(date)
        )
        if i==0:
            all_features = res
        else:
            all_features = pl.concat([all_features, res], how="vertical")
print(all_features)

#-----------------------------------------------------------------------------
# Save a plot of the cumulative performance of all analysts
#-----------------------------------------------------------------------------
if feature_engine.pnl_all_analysts is None:
    try:
        feature_engine.pnl_all_analysts = s3Utils.pull_parquet_file_from_s3(
            path="s3://alpha-in-analysts-storage/data/pnl_all_analysts.parquet",
            to_polars=True
        )
    except Exception as e:
        feature_engine._build_pnl_all_analysts()

pnl = feature_engine.pnl_all_analysts.to_pandas()
dates = pnl["date"].unique()
pnl["strat_ret"] = pnl["strat_ret"].fillna(0.0)
pnl["cum_ret"] = (
    (1 + pnl["strat_ret"])
    .groupby(pnl["analyst_id"])
    .cumprod()
    - 1
)

pivot = pnl.pivot(index="date", columns="analyst_id", values="cum_ret")
plt.figure(figsize=(12, 6))
plt.plot(pivot)
plt.title("Cumulative Performance overtime — All Analysts")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.savefig(config.ROOT_DIR/"outputs"/"figures"/"cum_ret_all_analysts.png", dpi=300, bbox_inches="tight")
plt.close()

#-----------------------------------------------------------------------------
# Meta Portfolio EW of all analysts
#-----------------------------------------------------------------------------
from alpha_in_analysts.backtester.strategies import BuyAndHold
from alpha_in_analysts.backtester.portfolio import EqualWeightingScheme
from alpha_in_analysts.backtester.backtest_pandas import Backtest
from alpha_in_analysts.backtester.analysis import PerformanceAnalyser
wide_ret = pnl.pivot(index="date", columns="analyst_id", values="strat_ret")
# Replace 0.0 by NaN
wide_ret = wide_ret.replace(0.0, np.nan)
# Strategy setup and benchmark
strategy = BuyAndHold(
    prices=pivot,
    returns=wide_ret,
)
strategy.compute_signals_values()
strategy.compute_signals()

# Portfolio level
ptf = EqualWeightingScheme(
    returns=wide_ret,
    aligned_returns=wide_ret.shift(-1),
    signals=strategy.signals,
    rebal_periods=1,
    portfolio_type="long_only"
)
ptf.compute_weights()
ptf.rebalance_portfolio()

# Backtesting
backtester = Backtest(
    returns=wide_ret.shift(-1),
    weights=ptf.rebalanced_weights,
    turnover=ptf.turnover,
    transaction_costs=10,
    strategy_name="LO EW Analysts"
)
backtester.run_backtest()
# Performance Analysis
analyzer = PerformanceAnalyser(
    portfolio_returns=backtester.cropped_portfolio_net_returns,
    bench_returns=backtester.cropped_portfolio_net_returns,
    freq="m",
    percentiles="",
    industries="",
    rebal_freq="1m"
)
metrics = analyzer.compute_metrics()
# Save Plot
analyzer.plot_cumulative_performance(
    saving_path=str(config.ROOT_DIR/"outputs"/"figures"/"cum_ret_lo_ew_analysts.png"),
)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# ML models
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# -----------------------------
# PARAMETERS
# -----------------------------
min_nb_periods_required = 24
validation_window = 12
forecast_horizon = 1

# -----------------------------
# DATA
# -----------------------------
if not isinstance(all_features, pd.DataFrame):
    data = all_features.to_pandas()
else:
    data = all_features
date_range = np.sort(data["date"].unique())
date_idx_range = [i for i, date in enumerate(date_range)]

# -----------------------------
# MODELS
# -----------------------------
# Models: ridge, lasso, random_forest
models = {"ridge":Ridge,
          "lasso":Lasso,
          "random_forest":RandomForestRegressor
          }
hyperparams_all_combinations = {
    "ridge": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
    "lasso": [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1.0}],
    "random_forest": [
        {"n_estimators": n, "max_depth": d, "random_state": 0}
        for n in [50, 100]
        for d in [5, 10]
    ]
}
nb_combinations = 0
for model in hyperparams_all_combinations:
    nb_combinations += len(hyperparams_all_combinations[model])

# -----------------------------
# METRIC
# -----------------------------
def rank_ic(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

# -----------------------------
# STORAGE (panel-safe)
# -----------------------------
OOS_PRED = {m: {} for m in models}
OOS_TRUE = {}

best_score_all_models_overtime = pd.DataFrame(
    index=date_range, columns=list(models.keys())
)

best_hyperparams_all_models_overtime = {
    m: pd.DataFrame(index=date_range, columns=list(hyperparams_all_combinations[m][0].keys()))
    for m in models
}

# -----------------------------
# WALK-FORWARD LOOP
# -----------------------------
start_idx = min_nb_periods_required + validation_window + forecast_horizon

for t in range(start_idx, len(date_range) - forecast_horizon):
    date_t = date_range[t]
    print(f"Training models for date {date_t} (t={t}) {t}/{len(date_range) - forecast_horizon - 1}")
    # Compute time to do one loop
    start = time.time()

    # -----------------------------
    # SPLITS
    # -----------------------------
    train_end = date_range[t - validation_window - forecast_horizon]
    val_end   = date_range[t - forecast_horizon]

    train_data = data[data["date"] <= train_end]
    val_data = data[(data["date"] > train_end) & (data["date"] <= val_end)]

    # -----------------------------
    # MODEL LOOP
    # -----------------------------
    for model_name, ModelClass in models.items():
        start_model = time.time()

        best_score = -np.inf
        best_hyperparams = None

        # -----------------------------
        # HYPERPARAMETER SELECTION
        # -----------------------------
        for nh, hyperparams in enumerate(hyperparams_all_combinations[model_name]):
            start_hyperparams = time.time()

            # Instantiate model
            model = ModelClass(**hyperparams)

            # Train
            X_train = train_data.drop(columns=["date", "analyst_id", "y"])
            y_train = train_data["y"]
            model.fit(X=X_train, y=y_train)

            # ---- validation IC per date
            ICs = []
            for d in np.sort(val_data["date"].unique()):
                X_val = val_data[val_data["date"] == d].drop(
                    columns=["date", "analyst_id", "y"]
                )
                y_val = val_data[val_data["date"] == d]["y"]

                y_hat = model.predict(X_val)

                if len(y_val) > 1:
                    # ic = rank_ic(y_val, y_hat)
                    ic = np.sqrt(mean_squared_error(y_val, y_hat))
                    if not np.isnan(ic):
                        ICs.append(ic)

            if len(ICs) == 0:
                continue

            score = np.mean(ICs)

            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams

            print(f"Loop finished for {model_name} / {hyperparams} ({nh+1}/{nb_combinations}) in: ", round((time.time() - start_hyperparams)/60, 4), "min")

        # -----------------------------
        # STORE VALIDATION RESULTS
        # -----------------------------
        best_score_all_models_overtime.loc[date_t, model_name] = best_score
        if best_hyperparams is not None:
            for k, v in best_hyperparams.items():
                best_hyperparams_all_models_overtime[model_name].loc[date_t, k] = v

        # -----------------------------
        # FINAL TRAINING
        # -----------------------------
        full_train = data[data["date"] <= val_end]

        model_final = ModelClass(**best_hyperparams)
        model_final.fit(
            X=full_train.drop(columns=["date", "analyst_id", "y"]),
            y=full_train["y"]
        )

        test_date = date_range[t]
        y_true_date = date_range[t]

        X_test = data[data["date"] == test_date].drop(
            columns=["date", "analyst_id", "y"]
        )
        y_hat = model_final.predict(X_test)

        OOS_PRED[model_name][test_date] = pd.Series(
            y_hat,
            index=data[data["date"] == test_date]["analyst_id"]
        )
        OOS_TRUE[test_date] = data[data["date"] == y_true_date].set_index(
            "analyst_id"
        )["y"]

        print(f"Loop finished for {model_name} in: ", round((time.time() - start_model)/60, 4), "min")

    print("Loop finished in: ", round((time.time() - start)/60, 4), "min")












