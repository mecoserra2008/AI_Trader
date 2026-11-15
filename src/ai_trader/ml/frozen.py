"""Frozen hourly volatility model (H-NN8 -> forward test H-NN10): fit once, save with a JSON card, load read-only.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.ml.frozen freeze [--as-of 2026-11-12T00:00Z]

freeze: per universe-L symbol, market_frame(symbol, '1h', as_of) from the lake (point in time), ml.hourly features,
complete rows with bar close <= as_of. Exactly one refit of the H-NN8 walk-forward protocol on all of it:
deterministic random 1/3 training subsample (hourly.train_subsample), models.val_split (last 15% of dates, purged 1 day),
fit-block Standardizer, XGBoost with the round-2 hyperparameters (data/models/ml/round2/search8.json; no new search),
early stopping on the validation block, validation-block QLIKE-optimal bias c. Forecast for bar t:
sigma^2 = s2_t exp(f(Z_t) + c). Writes data/models/ml/hourly_vol_v1/{model.json, card.json}.
The card holds the feature list, standardiser, bias, training window, hyperparameters, code hashes, library versions and
the walk-forward QLIKE measured in H-NN8. A frozen model is never refitted in place: a new fit is a new version directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.ml.hourly import HOURLY_FEATURES, HOURLY_GROUPS, complete_hourly, symbol_hourly, train_subsample

MODEL_ID = "hourly_vol_v1"
FREEZE_AS_OF = datetime(2026, 11, 12, tzinfo=UTC)


def model_dir(root: Path | None = None, model_id: str = MODEL_ID) -> Path:
    from ai_trader.core.config import load

    return (root or load().root / "data" / "models" / "ml") / model_id


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_head(cwd: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def panel_asof(symbols, as_of: datetime, lake=None) -> pl.DataFrame:
    """Complete hourly feature rows of every symbol whose bar has closed by as_of (bar open + 1h <= as_of)."""
    from ai_trader.research.frame import market_frame

    parts = []
    for s in symbols:
        f = market_frame(s, "1h", as_of, funding=False, oi=False, lake=lake)
        if f.height < 24 * 60:
            continue
        h = complete_hourly(symbol_hourly(f.select("event_ts", "ret", "sigma_risk_next")))
        h = h.with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC"))).filter(pl.col("event_ts") + pl.duration(hours=1) <= as_of)
        parts.append(h.with_columns(pl.lit(s).alias("symbol")))
    return pl.concat(parts).sort("event_ts", "symbol") if parts else pl.DataFrame()


def freeze(df: pl.DataFrame, out: Path, cfg: dict, *, as_of: datetime, extra_card: dict | None = None, nthread: int = 4) -> dict:
    """Fit the XGBoost hourly model on df (complete hourly rows) and save model.json + card.json under out."""
    from ai_trader.ml import models as md
    from ai_trader.ml.features import Standardizer

    import xgboost as xgb

    X = df.select(HOURLY_FEATURES).to_numpy().astype(float)
    y = df["y"].to_numpy().astype(float)
    T = df["event_ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    sub = np.flatnonzero(train_subsample(df.height))
    fl, vl = md.val_split(T[sub])
    fi, vi = sub[fl], sub[vl]
    st = Standardizer().fit(X[fi])
    m = md.fit_xgb(st.transform(X[fi]), y[fi], st.transform(X[vi]), y[vi], "vol", cfg=cfg, names=HOURLY_FEATURES, nthread=nthread)
    lv = np.log(df["s2"].to_numpy().astype(float)[vi])
    c = md.bias_const(df["r2"].to_numpy().astype(float)[vi], lv + m.predict(st.transform(X[vi])))
    out.mkdir(parents=True, exist_ok=True)
    m.booster.save_model(str(out / "model.json"))
    src = Path(__file__).resolve().parent
    card = {
        "model_id": out.name, "kind": "xgboost next-hour log squared return (H-NN8)", "frozen_at": as_of.isoformat(),
        "forecast": "sigma_next_hour^2 = s2 * exp(f(standardised features) + bias); s2 = sigma_risk_next of the last closed 1h bar squared",
        "baseline": "s2 * hour-of-week factor (features.seasonality.seasonal_factor, expanding; R-VOL-26 / H-NS4)",
        "features": list(HOURLY_FEATURES), "groups": {k: list(v) for k, v in HOURLY_GROUPS.items()},
        "standardizer": st.to_dict(), "bias": float(c), "best_iteration": int(m.best_iteration),
        "hyperparameters": {**md.XGB_DEFAULT, **cfg, "nthread": nthread, "seed": 0},
        "training": {"symbols": sorted(df["symbol"].unique().to_list()), "first_bar": str(df["event_ts"].min()), "last_bar": str(df["event_ts"].max()),
                     "rows_complete": df.height, "rows_fit": int(len(fi)), "rows_val": int(len(vi)), "subsample": "random 1/3, seed 20261112",
                     "val_first_bar": str(df["event_ts"][int(vi.min())]) if len(vi) else None},
        "code": {"git_head": _git_head(src), "hourly.py": _sha(src / "hourly.py"), "models.py": _sha(src / "models.py"), "frozen.py": _sha(src / "frozen.py")},
        "versions": {"xgboost": xgb.__version__, "numpy": np.__version__, "polars": pl.__version__},
        **(extra_card or {}),
    }
    (out / "card.json").write_text(json.dumps(card, indent=1, default=str), encoding="utf-8")
    return card


@dataclass(frozen=True)
class FrozenHourly:
    booster: object
    card: dict
    mu: np.ndarray
    sd: np.ndarray

    def margin(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        d = xgb.DMatrix(Z.astype(np.float32), feature_names=list(HOURLY_FEATURES))
        return self.booster.predict(d, iteration_range=(0, self.card["best_iteration"] + 1))

    def contribs(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        d = xgb.DMatrix(Z.astype(np.float32), feature_names=list(HOURLY_FEATURES))
        return self.booster.predict(d, iteration_range=(0, self.card["best_iteration"] + 1), pred_contribs=True)

    def variance(self, X: np.ndarray, s2: np.ndarray) -> np.ndarray:
        return np.asarray(s2, dtype=float) * np.exp(self.margin(X) + self.card["bias"])


def load_frozen(path: Path | None = None) -> FrozenHourly:
    import xgboost as xgb

    p = Path(path) if path is not None else model_dir()
    card = json.loads((p / "card.json").read_text(encoding="utf-8"))
    if list(card["features"]) != list(HOURLY_FEATURES):
        raise ValueError(f"{p}: feature list differs from ml.hourly.HOURLY_FEATURES")
    b = xgb.Booster()
    b.load_model(str(p / "model.json"))
    b.set_param({"nthread": 1})
    return FrozenHourly(b, card, np.asarray(card["standardizer"]["mu"], dtype=float), np.asarray(card["standardizer"]["sd"], dtype=float))


def _main() -> None:
    from ai_trader.core.config import load
    from ai_trader.research.trend_zoo_study import UNIVERSE

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("freeze",))
    ap.add_argument("--as-of", default=FREEZE_AS_OF.isoformat())
    a = ap.parse_args()
    as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC)
    out = model_dir()
    if (out / "card.json").exists():
        raise SystemExit(f"{out} already exists: a frozen model is never overwritten (use a new version id)")
    root = load().root
    s8 = json.loads((root / "data" / "models" / "ml" / "round2" / "search8.json").read_text(encoding="utf-8"))
    rep = json.loads((root / "research" / "reports" / "ml_signals_round2_2026-11-12.json").read_text(encoding="utf-8"))
    measured = {s: {k: rep["H-NN8"][s][f"{k}_vs_champ"] for k in ("xgb", "mlp", "lin")} for s in ("llm_in_sample", "clean_holdout")}
    df = panel_asof(UNIVERSE, as_of)
    card = freeze(df, out, dict(s8["xgb"]["best"]), as_of=as_of, extra_card={
        "walk_forward_qlike_vs_baseline_H-NN8": measured, "search": "data/models/ml/round2/search8.json (pre-2021-07 rows; frozen choice)",
        "forward_test": "H-NN10: hours after 2026-11-12, NW t by date, >= 60 forward days"})
    print(json.dumps({k: card[k] for k in ("model_id", "best_iteration", "bias", "training", "hyperparameters")}, indent=1, default=str))


if __name__ == "__main__":
    _main()
