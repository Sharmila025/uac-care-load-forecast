"""
engine.py : forecasting core for the UAC Care Load & Placement Demand project.

Everything here is UI-independent (no Streamlit), so it can be unit-tested,
reused in a notebook, or imported by the dashboard (app.py).

Pipeline
    load_raw -> prepare_daily -> build_features -> models -> walk_forward_backtest
                                                         -> forecast_future -> KPIs
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")

# statsmodels is optional: if missing, the statistical models are skipped.
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

# --------------------------------------------------------------------------- #
# 1. Loading and time-series preparation
# --------------------------------------------------------------------------- #
COLUMN_MAP = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "apprehended",
    "Children apprehended and placed in CBP custody": "apprehended",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transferred",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharged",
}
FLOW_COLS = ["apprehended", "cbp_custody", "transferred", "hhs_care", "discharged"]
TARGETS = {
    "hhs_care": "Children in HHS care",
    "discharged": "Daily discharges (placements)",
}


def load_raw(source) -> pd.DataFrame:
    """Read the HHS CSV (path, file-like or bytes) and clean it.

    Handles: blank trailing rows, thousands separators stored as text
    ("2,484"), "Month DD, YYYY" dates and the footnote asterisk in a header.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    missing = [c for c in ["date"] + FLOW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = df[["date"] + FLOW_COLS].dropna(how="all")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in FLOW_COLS:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    return df


def prepare_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Continuous daily index; missing days filled by time interpolation.

    Adds `imputed` (True for days that were not in the source file) and
    `net_pressure` = transferred - discharged.
    """
    full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    daily = raw.reindex(full_idx)
    daily.index.name = "date"
    daily["imputed"] = daily["hhs_care"].isna()
    daily[FLOW_COLS] = daily[FLOW_COLS].interpolate(method="time").ffill().bfill()
    daily["net_pressure"] = daily["transferred"] - daily["discharged"]
    return daily


def decompose(series: pd.Series, period: int = 7) -> pd.DataFrame:
    """Classical additive decomposition (trend / seasonal / residual)."""
    trend = series.rolling(period, center=True, min_periods=period // 2 + 1).mean()
    detr = series - trend
    seas_profile = detr.groupby(series.index.dayofweek).mean()
    seas_profile -= seas_profile.mean()
    seasonal = pd.Series(series.index.dayofweek.map(seas_profile).values, index=series.index)
    resid = series - trend - seasonal
    return pd.DataFrame({"observed": series, "trend": trend, "seasonal": seasonal, "residual": resid})


# --------------------------------------------------------------------------- #
# 2. Feature engineering (all features use information up to t-1 only)
# --------------------------------------------------------------------------- #
_HOLIDAYS = USFederalHolidayCalendar().holidays(start="2022-01-01", end="2030-12-31")


def build_features(w: pd.DataFrame, target: str) -> pd.DataFrame:
    y = w[target]
    X = pd.DataFrame(index=w.index)
    for lag in (1, 7, 14):
        X[f"lag_{lag}"] = y.shift(lag)
    for k in (7, 14):
        r = y.shift(1).rolling(k)
        X[f"roll_mean_{k}"] = r.mean()
        X[f"roll_var_{k}"] = r.var()
    net = w["transferred"] - w["discharged"]
    X["net_lag_1"] = net.shift(1)
    X["net_roll_7"] = net.shift(1).rolling(7).mean()
    X["net_roll_14"] = net.shift(1).rolling(14).mean()
    X["xfer_roll_7"] = w["transferred"].shift(1).rolling(7).mean()
    X["dis_roll_7"] = w["discharged"].shift(1).rolling(7).mean()
    X["appr_roll_7"] = w["apprehended"].shift(1).rolling(7).mean()
    X["cbp_lag_1"] = w["cbp_custody"].shift(1)
    X["dow"] = w.index.dayofweek
    X["month"] = w.index.month
    near_hol = pd.Series(False, index=w.index)
    for off in (-1, 0, 1):
        near_hol |= (w.index + pd.Timedelta(days=off)).isin(_HOLIDAYS)
    X["holiday_proxy"] = near_hol.astype(int).values
    return X


# --------------------------------------------------------------------------- #
# 3. Models. Each returns a numpy array of `h` forecasts for the days after the
#    last date in `train`.
# --------------------------------------------------------------------------- #
def f_naive(train: pd.DataFrame, target: str, h: int) -> np.ndarray:
    return np.repeat(train[target].iloc[-1], h)


def f_moving_average(train: pd.DataFrame, target: str, h: int, window: int = 7) -> np.ndarray:
    return np.repeat(train[target].iloc[-window:].mean(), h)


def f_ets(train: pd.DataFrame, target: str, h: int) -> np.ndarray:
    y = train[target].astype(float)
    y = y.asfreq("D")
    try:
        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
            initialization_method="estimated",
        ).fit()
        out = np.asarray(fit.forecast(h))
    except Exception:
        out = f_naive(train, target, h)
    return out


def f_sarima(train: pd.DataFrame, target: str, h: int) -> np.ndarray:
    y = train[target].astype(float).asfreq("D")
    try:
        fit = SARIMAX(
            y, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=60)
        out = np.asarray(fit.forecast(h))
    except Exception:
        out = f_naive(train, target, h)
    return out


def _ml_factory(kind: str):
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, max_features=0.6, n_jobs=-1, random_state=42
        )
    return GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )


def f_ml(train: pd.DataFrame, target: str, h: int, kind: str) -> np.ndarray:
    """Recursive multi-step forecast with a tree ensemble.

    * hhs_care   -> the model predicts the *daily change*, then we integrate
                    (a stock variable; predicting change avoids a tree model
                    being unable to extrapolate outside its training range).
    * discharged -> the model predicts the level directly.
    Future values of the other flow columns are held at their trailing 7-day mean.
    """
    w = train[FLOW_COLS].copy()
    X = build_features(w, target)
    y = w[target].diff() if target == "hhs_care" else w[target]
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    model = _ml_factory(kind).fit(data.drop(columns="y"), data["y"])
    cols = list(data.drop(columns="y").columns)

    w = w.copy()
    out = []
    for step in range(1, h + 1):
        d = w.index[-1] + pd.Timedelta(days=1)
        new = w.iloc[-7:].mean()  # trailing-mean assumption for exogenous flows
        w.loc[d] = new
        w.loc[d, target] = np.nan
        feats = build_features(w.tail(40), target).iloc[[-1]][cols]
        pred = float(model.predict(feats)[0])
        if target == "hhs_care":
            val = w[target].iloc[-2] + pred
        else:
            val = max(pred, 0.0)
        w.loc[d, target] = val
        out.append(val)
    return np.array(out)


MODEL_REGISTRY = {
    "Naive persistence": ("baseline", lambda tr, t, h: f_naive(tr, t, h)),
    "Moving average (7d)": ("baseline", lambda tr, t, h: f_moving_average(tr, t, h)),
    "Exponential smoothing": ("statistical", lambda tr, t, h: f_ets(tr, t, h)),
    "SARIMA": ("statistical", lambda tr, t, h: f_sarima(tr, t, h)),
    "Random Forest": ("machine learning", lambda tr, t, h: f_ml(tr, t, h, "rf")),
    "Gradient Boosting": ("machine learning", lambda tr, t, h: f_ml(tr, t, h, "gb")),
}


def available_models() -> list[str]:
    names = list(MODEL_REGISTRY)
    if not HAS_STATSMODELS:
        names = [n for n in names if MODEL_REGISTRY[n][0] != "statistical"]
    return names


# --------------------------------------------------------------------------- #
# 4. Walk-forward validation and metrics
# --------------------------------------------------------------------------- #
@dataclass
class Backtest:
    target: str
    horizon: int
    origins: list
    actual: np.ndarray                 # (n_origins, horizon)
    preds: dict                        # model -> (n_origins, horizon)

    def errors(self, model):
        return self.preds[model] - self.actual


def walk_forward_backtest(daily, target, models, horizon=30, n_origins=14, step=30,
                          progress=None) -> Backtest:
    """Expanding-window, strictly time-ordered validation.

    For each forecast origin we train only on data up to that day and forecast
    the next `horizon` days, then compare with what really happened.
    """
    n = len(daily)
    last_origin = n - horizon
    origins_idx = [last_origin - i * step for i in range(n_origins)][::-1]
    origins_idx = [i for i in origins_idx if i >= 120]
    actual = np.array([daily[target].iloc[i: i + horizon].values for i in origins_idx])
    preds = {m: [] for m in models}
    total = len(origins_idx) * len(models)
    done = 0
    for i in origins_idx:
        train = daily.iloc[:i]
        for m in models:
            preds[m].append(MODEL_REGISTRY[m][1](train, target, horizon))
            done += 1
            if progress:
                progress(done / total, f"Walk-forward: {m}")
    return Backtest(
        target=target, horizon=horizon,
        origins=[daily.index[i - 1] for i in origins_idx],
        actual=actual, preds={m: np.array(v) for m, v in preds.items()},
    )


def metrics_by_horizon(bt: Backtest, model: str) -> pd.DataFrame:
    e = bt.errors(model)
    a = np.maximum(np.abs(bt.actual), 1.0)
    return pd.DataFrame({
        "horizon": np.arange(1, bt.horizon + 1),
        "MAE": np.abs(e).mean(0),
        "RMSE": np.sqrt((e ** 2).mean(0)),
        "MAPE": (np.abs(e) / a).mean(0) * 100,
    })


def summary_table(bt: Backtest, horizon: int) -> pd.DataFrame:
    """Per-model scorecard averaged over horizons 1..horizon.

    Stability index : 100 / (1 + CV) of the per-fold MAE. 100 means the error
                      is identical in every walk-forward fold; lower means the
                      model's quality swings between regimes.
    Robustness      : % of walk-forward folds in which the model's MAE is no
                      worse than naive persistence (n/a for naive itself).
    """
    naive_fold = np.abs(bt.errors("Naive persistence")[:, :horizon]).mean(1) if "Naive persistence" in bt.preds else None
    rows = []
    for m in bt.preds:
        mh = metrics_by_horizon(bt, m)
        s = mh[mh.horizon <= horizon]
        fold_mae = np.abs(bt.errors(m)[:, :horizon]).mean(1)
        cv = fold_mae.std() / fold_mae.mean() if fold_mae.mean() > 0 else 0
        if naive_fold is None or m == "Naive persistence":
            rob = np.nan
        else:
            rob = 100 * float((fold_mae <= naive_fold).mean())
        rows.append({
            "Model": m,
            "MAE": s.MAE.mean(),
            "RMSE": s.RMSE.mean(),
            "MAPE %": s.MAPE.mean(),
            "Accuracy %": max(0.0, 100 - s.MAPE.mean()),
            "Stability index": 100 / (1 + cv),
            "Robustness": rob,
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def sigma_by_horizon(bt: Backtest, model: str) -> np.ndarray:
    """Relative forecast-error std by horizon (error as a fraction of the level).

    Using relative errors makes the interval scale with the current load, so a
    volatile high-volume period does not inflate bands during a calm low-volume
    one. Forced non-decreasing in horizon.
    """
    rel = bt.errors(model) / np.maximum(np.abs(bt.actual), 1.0)
    sig = np.sqrt((rel ** 2).mean(0))
    return np.maximum.accumulate(np.maximum(sig, 0.002))


# --------------------------------------------------------------------------- #
# 5. Production forecast + uncertainty + KPIs
# --------------------------------------------------------------------------- #
def forecast_future(daily, target, model, horizon, sigma=None, level=0.95):
    """Forecast beyond the last observed day. `sigma` = relative error by horizon
    (from sigma_by_horizon). Returns a DataFrame with confidence bounds."""
    point = MODEL_REGISTRY[model][1](daily, target, horizon)
    point = np.maximum(point, 0)
    idx = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    out = pd.DataFrame({"forecast": point}, index=idx)
    if sigma is not None:
        z = norm.ppf(0.5 + level / 2)
        s = np.asarray(sigma)[:horizon] * np.maximum(point, 1.0)
        out["lower"] = np.maximum(point - z * s, 0)
        out["upper"] = point + z * s
        out["sigma"] = s
    return out


def breach_probability(fc: pd.DataFrame, cap: float) -> pd.Series:
    """P(level > cap) per day under a normal error model."""
    return pd.Series(1 - norm.cdf((cap - fc["forecast"]) / fc["sigma"]), index=fc.index)


def surge_lead_time(fc: pd.DataFrame, cap: float):
    """Days until the point forecast first exceeds the threshold (None = no breach)."""
    hit = np.where(fc["forecast"].values > cap)[0]
    return int(hit[0] + 1) if len(hit) else None


def scenario_adjust(fc: pd.DataFrame, daily, transfer_pct=0.0, discharge_pct=0.0):
    """Shift a care-load forecast by a what-if change in daily flows.

    transfer_pct   : +% on expected daily transfers into HHS
    discharge_pct  : +% on expected daily discharges
    The extra net flow accumulates day by day on top of the base forecast.
    """
    base_in = daily["transferred"].iloc[-14:].mean()
    base_out = daily["discharged"].iloc[-14:].mean()
    delta = base_in * transfer_pct / 100 - base_out * discharge_pct / 100
    adj = fc.copy()
    shift = delta * np.arange(1, len(fc) + 1)
    for c in ("forecast", "lower", "upper"):
        if c in adj:
            adj[c] = np.maximum(adj[c] + shift, 0)
    return adj, delta


def stress_status(daily, fc: pd.DataFrame, cap: float):
    """Traffic-light early warning combining breach risk and net inflow trend."""
    p_max = float(breach_probability(fc, cap).max()) if "sigma" in fc else 0.0
    net7 = daily["net_pressure"].iloc[-7:].mean()
    if p_max >= 0.5 or (net7 > 0 and fc["forecast"].iloc[-1] > cap):
        return "Critical", p_max, net7
    if p_max >= 0.2 or net7 > 0.1 * max(daily["discharged"].iloc[-7:].mean(), 1):
        return "Watch", p_max, net7
    return "Stable", p_max, net7


def add_interval(fc: pd.DataFrame, rel_sigma, level=0.95) -> pd.DataFrame:
    """Attach lower/upper/sigma columns to a point forecast frame."""
    out = fc[["forecast"]].copy()
    z = norm.ppf(0.5 + level / 2)
    s = np.asarray(rel_sigma)[: len(out)] * np.maximum(out["forecast"].values, 1.0)
    out["sigma"] = s
    out["lower"] = np.maximum(out["forecast"] - z * s, 0)
    out["upper"] = out["forecast"] + z * s
    return out
