"""WS12 ML benchmark, deep part: small sequence networks vs classical forecasts (specification program 2026-11-27).

Research only; deterministic given the seed. Registered as H-WS12-1..5 before any fit (research/hypotheses_pending/WS12.yaml,
diary "WS12 design frozen before results").

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.ws12_deep build
    ... train --arch gru|tcn|mlp --refit R1|R2        (lr choice on the inner validation block with seed 0, then seeds 1-2)
    ... classical                                       (log-HAR-t, base-t, empirical quantiles; per asset, per refit)
    ... evaluate [--record]                             (scores, DM, MCS, seed spread; report tables JSON)

Data (point-in-time; bars with known_ts <= AS_OF; 1h Bybit last-price perps BTC ETH SOL XRP DOGE BNB):
  grid = ml.hourly.symbol_hourly (regular 1h grid, a return only when the previous bar exists). Decision time t = open of
  bar t. s2_t = robust_bc base variance known at the open (sigma_risk_next of bar t-1, squared).
  Per-step inputs for bar j <= t-1 (STEP_FEATURES): z = r_j / s_pre_j (clip +-10), lu = log max(z^2, 1e-4), lrange =
  log max(ln(H/L)^2 / (4 ln2 s_pre_j^2), 1e-4), lsig = log s2_{j+1} (made relative to the last step inside the batch, so
  the window carries the vol path, not the level), lturn = log turnover - log median turnover of the previous 720 bars,
  fund = last settled funding rate (bp per 8h, clip +-20), doi = 24-bar change of log OI (4h interval, as-of, clip +-1),
  calendar of bar j+1 (hour / weekday sin-cos, funding-hour flag), lfacn = log hour-of-week factor of bar j+1 (H-NS4,
  causal: uses u of bars <= j).
  Static at t (STATIC_FEATURES): lsig_rel1y = log s2_t - mean log s2 over the previous 8760 bars (>= 2000), har_d/w/m
  (ml.hourly), lfac_n4 / lfac_n24 = log mean factor over hours t..t+3 / t..t+23 from the profile known at t (uses only
  bars < t; features.seasonality._shape, same kappa / kernel / cap / floor).
  Targets: y_h = sum_{k<h} r_{t+k} / sqrt(h s2_t); v_h = sum r^2 / (h s2_t); h in (1, 4, 24); all h bars present.
  Window: 72 bars t-72..t-1, all inputs finite. Label end = t + 24 h for every row (multi-task, longest horizon).
Models (pooled over assets, no asset id, Standardizer on the fit block, inputs clipped +-5):
  gru  1-layer GRU (d 32) -> last state;  tcn  1x1 input projection + 4 residual blocks of two causal dilated convs (k 3,
  dilation 1/2/4/8, receptive field 61) -> last step;  mlp  2 x Linear(d) on the 19 H-NN8 HOURLY_FEATURES at t.
  Head: [enc, static] -> Linear 32 -> GELU -> per horizon (mu, scale, df, log F): Student-t with df = 2.05 + softplus
  (cap 100) for y_h, and a log-RV forecast F_h for v_h. Loss = mean over horizons of (t NLL + QLIKE(v, F)), QLIKE =
  v / F + log F. AdamW (wd 1e-4), batch 256, epoch = 16384 random fit origins, max 40 epochs, patience 6 on a fixed
  subsample (<= 16384) of the inner validation block, grad clip 1, dropout 0.1 on the encoder output.
Walk: refits R1 = 2025-07-01 (test to 2026-06-30: llm_in_sample) and R2 = 2026-07-01 (test to the end: clean_holdout);
  training rows label_end <= refit - 1 day; inner validation = last 90 days of training dates, fit rows end 1 day before.
  lr in {1e-3, 3e-3} chosen by inner-validation loss with seed 0; seeds 1, 2 with the chosen lr. D* = gru or tcn by R1
  inner-validation loss of the chosen config (fixed before scoring).
Classical (per asset, fitted on the same training rows): base-t: F_h = exp(lfac_nh) (robust sigma x hour-of-week),
  Student-t (MLE) on y_h / sqrt(F_h); log-HAR-t: OLS of log max(v_h, 1e-6) on [1, lsig2, lfac_nh, har_d, har_w, har_m],
  QLIKE bias constant (ml.models.bias_const), Student-t on y_h / sqrt(F_h); emp: quantiles of the trailing 720 / 180 / 365
  non-overlapping y_h (labels ended by t), RV forecast = their mean v_h.
Scores (vol-standardised units) at non-overlapping origins (1h every hour, 4h at hour % 4 == 0, 24h at 00 UTC): pinball
  over TAUS, CRPS by the 99-quantile approximation, log score (density models; seed ensemble = equal mixture), QLIKE.
  Seed ensemble: quantiles averaged (Vincentisation), F averaged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load

TAG = "2026-11-27"
AS_OF = datetime(2026, 11, 26, tzinfo=UTC)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT")
HS = (1, 4, 24)
L = 72
STEP_FEATURES = ("z", "lu", "lrange", "lsig", "lturn", "fund", "doi", "hod_sin", "hod_cos", "dow_sin_n", "dow_cos_n", "fund_next", "lfacn")
STATIC_FEATURES = ("lsig_rel1y", "har_d", "har_w", "har_m", "lfac_n4", "lfac_n24")
LSIG = STEP_FEATURES.index("lsig")
TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TAU99 = tuple(k / 100 for k in range(1, 100))
REFITS = {"R1": (datetime(2025, 7, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)),
          "R2": (datetime(2026, 7, 1, tzinfo=UTC), AS_OF)}
EMBARGO = timedelta(days=1)
VAL_DAYS = 90
LRS = (1e-3, 3e-3)
SEEDS = (0, 1, 2)
D = 32
BATCH, EPOCH_N, MAX_EP, PATIENCE, VAL_N = 256, 16384, 40, 6, 16384
EMP_N = {1: 720, 4: 180, 24: 365}


def root() -> Path:
    p = load().root / "data" / "research" / "ws12"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _np_us(s: pl.Series) -> np.ndarray:
    return s.dt.replace_time_zone(None).cast(pl.Datetime("us")).to_numpy()


# ------------------------------------------------------------------ data


def factor_ahead(how: np.ndarray, u: np.ndarray, hs=(4, 24), kappa=20.0, bandwidth=1, cap=50.0, floor=0.1, min_weeks=8) -> dict[int, np.ndarray]:
    """log mean hour-of-week factor over hours t..t+h-1 from the profile known at t (u of bars < t only)."""
    from ai_trader.features.seasonality import HOW, _kernel, _shape

    n = len(u)
    kern = _kernel(bandwidth)
    sums, counts = np.zeros(HOW), np.zeros(HOW)
    out = {h: np.full(n, np.nan) for h in hs}
    seen, need, prof = 0, min_weeks * HOW, None
    for t in range(n):
        if seen >= need:
            if prof is None:
                prof = _shape(sums, counts, kappa, kern, floor)
                ext = np.concatenate([prof, prof])
            for h in hs:
                k = how[t]
                out[h][t] = math.log(ext[k:k + h].mean())
        x = u[t]
        if np.isfinite(x):
            k = how[t]
            sums[k] += min(max(x, 0.0), cap)
            counts[k] += 1
            seen += 1
            prof = None
    return out


def build_symbol(sym: str) -> pl.DataFrame:
    from ai_trader.features import seasonality as sz
    from ai_trader.ml.hourly import symbol_hourly
    from ai_trader.research.frame import market_frame

    f = market_frame(sym, "1h", AS_OF, funding=True, oi=True, oi_interval="4h")
    h = symbol_hourly(f)
    f2 = f.select("event_ts", "ret", "high", "low", "turnover", "funding_rate", "open_interest", "sigma_risk_next").unique("event_ts", keep="last")
    g = h.join(f2, on="event_ts", how="left").sort("event_ts")
    n = g.height
    ok_r = g["r2"].is_not_null().to_numpy()
    r = np.where(ok_r, g["ret"].cast(pl.Float64).fill_null(np.nan).to_numpy(), np.nan)
    s2 = g["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    s2n = g["sigma_risk_next"].cast(pl.Float64).fill_null(np.nan).to_numpy() ** 2
    hi, lo = (g[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in ("high", "low"))
    turn = g["turnover"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    fund = g["funding_rate"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    oi = g["open_interest"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    ts = g["event_ts"]
    with np.errstate(all="ignore"):
        z = np.clip(r / np.sqrt(s2), -10, 10)
        lu = np.log(np.maximum(z * z, 1e-4))
        lrange = np.log(np.maximum(np.log(hi / lo) ** 2 / (4 * math.log(2) * s2), 1e-4))
        lsig = np.log(s2n)
        lt = np.log(np.where(turn > 0, turn, np.nan))
        med = pl.Series(lt).fill_nan(None).shift(1).rolling_median(720, min_samples=240).to_numpy().astype(float)
        lturn = np.clip(lt - med, -5, 5)
        fundbp = np.clip(fund * 1e4, -20, 20)
        loi = np.log(np.where(oi > 0, oi, np.nan))
        doi = np.clip(loi - np.r_[np.full(24, np.nan), loi[:-24]], -1, 1)
        ls2 = np.log(s2)
        m1y = pl.Series(ls2).fill_nan(None).shift(1).rolling_mean(8760, min_samples=2000).to_numpy().astype(float)
        lsig_rel1y = ls2 - m1y
    nxt = ts + timedelta(hours=1)
    hr = nxt.dt.hour().to_numpy().astype(float)
    wd = nxt.dt.weekday().to_numpy().astype(float)
    fac = g["fac"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    lfacn = np.r_[np.log(fac[1:]), np.nan]
    how = sz.hour_of_week(ts)
    with np.errstate(all="ignore"):
        u = r * r / s2
    fa = factor_ahead(how, u)
    cols = {
        "z": z, "lu": lu, "lrange": lrange, "lsig": lsig, "lturn": lturn, "fund": fundbp, "doi": doi,
        "hod_sin": np.sin(2 * np.pi * hr / 24), "hod_cos": np.cos(2 * np.pi * hr / 24),
        "dow_sin_n": np.sin(2 * np.pi * (wd - 1) / 7), "dow_cos_n": np.cos(2 * np.pi * (wd - 1) / 7),
        "fund_next": np.isin(hr, (0, 8, 16)).astype(float), "lfacn": lfacn,
        "lsig_rel1y": lsig_rel1y, "lfac_n4": fa[4], "lfac_n24": fa[24], "r": r,
    }
    # targets
    rr = np.where(np.isfinite(r), r, np.nan)
    for hh in HS:
        sw = np.full(n, np.nan)
        s2w = np.full(n, np.nan)
        if n >= hh:
            win = np.lib.stride_tricks.sliding_window_view(rr, hh)  # win[t] = r[t..t+hh-1]
            sw[: n - hh + 1] = win.sum(axis=1)
            s2w[: n - hh + 1] = (win * win).sum(axis=1)
        with np.errstate(all="ignore"):
            cols[f"y{hh}"] = sw / np.sqrt(hh * s2)
            cols[f"v{hh}"] = s2w / (hh * s2)
    out = g.select("event_ts", "lsig2", "lfac", "fund_after", "fund_before", "lu_1", "lu_2", "lu_3", "lu_6", "lu_12", "lu_24",
                   "lu_m6", "lu_m24", "jump24", "since_jump", "har_d", "har_w", "har_m", "dow_sin", "dow_cos", "s2").with_columns(
        *[pl.Series(k, np.where(np.isfinite(v), v, np.nan)).cast(pl.Float32) for k, v in cols.items()], pl.lit(sym).alias("symbol"))
    return out.with_columns(pl.col(pl.Float32).fill_nan(None), pl.col(pl.Float64).fill_nan(None))


def build() -> None:
    d = root() / "cache"
    d.mkdir(exist_ok=True)
    for s in SYMBOLS:
        p = d / f"{s}.parquet"
        if p.is_file():
            continue
        t0 = time.time()
        df = build_symbol(s)
        df.write_parquet(p, compression="zstd")
        print(f"{s}: {df.height} rows, {time.time() - t0:.0f}s", flush=True)


def panel() -> dict:
    """Concatenated arrays with L NaN rows between assets (windows never cross assets)."""
    from ai_trader.ml.hourly import HOURLY_FEATURES

    parts = []
    for s in SYMBOLS:
        df = pl.read_parquet(root() / "cache" / f"{s}.parquet")
        parts.append(df)
    P, S, M, Y, V, T, A, R, S2 = [], [], [], [], [], [], [], [], []
    for ai, df in enumerate(parts):
        n = df.height
        pad = lambda a, w: np.vstack([np.full((L, w), np.nan, np.float32), a])  # noqa: E731
        P.append(pad(df.select(STEP_FEATURES).to_numpy().astype(np.float32), len(STEP_FEATURES)))
        S.append(pad(df.select(STATIC_FEATURES).to_numpy().astype(np.float32), len(STATIC_FEATURES)))
        M.append(pad(df.select(HOURLY_FEATURES).to_numpy().astype(np.float32), len(HOURLY_FEATURES)))
        Y.append(pad(df.select([f"y{h}" for h in HS]).to_numpy().astype(np.float32), 3))
        V.append(pad(df.select([f"v{h}" for h in HS]).to_numpy().astype(np.float32), 3))
        T.append(np.r_[np.full(L, np.datetime64("NaT"), "datetime64[us]"), _np_us(df["event_ts"])])
        A.append(np.r_[np.full(L, -1), np.full(n, ai)])
        R.append(np.r_[np.full(L, np.nan), df["r"].cast(pl.Float64).fill_null(np.nan).to_numpy()])
        S2.append(np.r_[np.full(L, np.nan), df["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy()])
    P, S, M, Y, V = (np.vstack(x) for x in (P, S, M, Y, V))
    T, A, R, S2 = (np.concatenate(x) for x in (T, A, R, S2))
    stepok = np.isfinite(P).all(axis=1)
    bad = (~stepok).astype(np.int64)
    cs = np.r_[0, np.cumsum(bad)]
    idx = np.arange(len(P))
    winok = np.zeros(len(P), bool)
    winok[L:] = (cs[idx[L:]] - cs[idx[L:] - L]) == 0  # rows t-L..t-1 all finite
    ok = winok & np.isfinite(S).all(axis=1) & np.isfinite(M).all(axis=1) & np.isfinite(Y).all(axis=1) & np.isfinite(V).all(axis=1) & (A >= 0)
    hour = np.where(A >= 0, (T.astype("datetime64[h]").astype(np.int64) % 24), -1)
    return {"P": P, "S": S, "M": M, "Y": Y, "V": V, "T": T, "A": A, "ok": ok, "hour": hour, "r": R, "s2": S2}


def _us(d: datetime) -> np.datetime64:
    return np.datetime64(d.astimezone(UTC).replace(tzinfo=None), "us")


def split_rows(pn: dict, refit: str) -> dict:
    a, b = REFITS[refit]
    T, ok = pn["T"], pn["ok"]
    le = T + np.timedelta64(24, "h")
    cut = _us(a - EMBARGO)
    tr = ok & (le <= cut)
    days = np.unique(T[tr].astype("datetime64[D]"))
    v0 = days[-VAL_DAYS].astype("datetime64[us]")
    va = tr & (T >= v0)
    fi = tr & (le <= v0 - np.timedelta64(1, "D"))
    end = _us(b)
    te = ok & (T >= _us(a)) & (T < end) & (T + np.timedelta64(24, "h") <= _us(AS_OF))
    return {"fit": np.flatnonzero(fi), "val": np.flatnonzero(va), "train": np.flatnonzero(tr), "test": np.flatnonzero(te)}


# ------------------------------------------------------------------ networks


def _torch():
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", 2)))
    return torch


def make_net(arch: str, n_step: int, n_static: int, n_mlp: int, d: int = D):
    torch = _torch()
    nn = torch.nn

    class Causal(nn.Module):
        def __init__(self, c, k, dil):
            super().__init__()
            self.pad = (k - 1) * dil
            self.conv = nn.Conv1d(c, c, k, dilation=dil)

        def forward(self, x):
            return self.conv(nn.functional.pad(x, (self.pad, 0)))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_step, d, 1)
            self.blocks = nn.ModuleList([nn.Sequential(Causal(d, 3, 2 ** b), nn.GELU(), Causal(d, 3, 2 ** b), nn.GELU()) for b in range(4)])

        def forward(self, x):  # x (B, L, F)
            h = self.inp(x.transpose(1, 2))
            for blk in self.blocks:
                h = h + blk(h)
            return h[:, :, -1]

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_step, d, batch_first=True)

        def forward(self, x):
            _, hN = self.rnn(x)
            return hN[-1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_mlp, d), nn.GELU(), nn.Linear(d, d), nn.GELU())

        def forward(self, x):
            return self.net(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            self.enc = {"gru": GRU, "tcn": TCN, "mlp": MLP}[arch]()
            self.drop = nn.Dropout(0.1)
            self.head = nn.Sequential(nn.Linear(d + n_static, 32), nn.GELU(), nn.Linear(32, 4 * len(HS)))

        def forward(self, x, s):
            h = self.drop(self.enc(x))
            o = self.head(torch.cat([h, s], dim=1)).view(-1, len(HS), 4)
            mu = o[..., 0]
            scale = nn.functional.softplus(o[..., 1] + 0.5) + 1e-3
            df = torch.clamp(2.05 + nn.functional.softplus(o[..., 2] + 1.5), max=100.0)
            logF = o[..., 3]
            return mu, scale, df, logF

    return Net()


def t_nll_torch(y, mu, scale, df):
    torch = _torch()
    zz = (y - mu) / scale
    return -(torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2) - 0.5 * torch.log(df * math.pi) - torch.log(scale)
             - (df + 1) / 2 * torch.log1p(zz * zz / df))


class Batcher:
    """Builds (x, s, y, v) tensors from row indices of the panel with the fit-block standardiser."""

    def __init__(self, pn: dict, arch: str, fit_rows: np.ndarray, rng: np.random.Generator):
        self.pn, self.arch = pn, arch
        k = 50000 if arch == "mlp" else 4000
        sub = fit_rows if len(fit_rows) <= k else rng.choice(fit_rows, k, replace=False)
        if arch == "mlp":
            X = pn["M"][sub]
        else:
            W = self._win(sub)
            X = W.reshape(-1, W.shape[2])
        self.mx, self.sx = np.nanmean(X, axis=0), np.nanstd(X, axis=0) + 1e-6
        Sx = pn["S"][sub]
        self.ms, self.ss = Sx.mean(axis=0), Sx.std(axis=0) + 1e-6

    def _win(self, rows: np.ndarray) -> np.ndarray:
        W = self.pn["P"][rows[:, None] + np.arange(-L, 0)[None, :]].copy()  # (B, L, F)
        W[:, :, LSIG] -= W[:, -1:, LSIG]
        return W

    def state(self) -> dict:
        return {k: getattr(self, k).astype(float).tolist() for k in ("mx", "sx", "ms", "ss")}

    def __call__(self, rows: np.ndarray):
        torch = _torch()
        if self.arch == "mlp":
            X = (self.pn["M"][rows] - self.mx) / self.sx
        else:
            X = (self._win(rows) - self.mx) / self.sx
        Sx = (self.pn["S"][rows] - self.ms) / self.ss
        f = lambda a: torch.from_numpy(np.clip(a, -5, 5).astype(np.float32))  # noqa: E731
        return f(X), f(Sx), torch.from_numpy(self.pn["Y"][rows]), torch.from_numpy(self.pn["V"][rows])


def loss_fn(net, batch):
    torch = _torch()
    x, s, y, v = batch
    mu, scale, df, logF = net(x, s)
    nll = t_nll_torch(y, mu, scale, df)
    ql = v * torch.exp(-logF) + logF
    return (nll + ql).mean(), nll.mean(dim=0), ql.mean(dim=0)


def train_one(pn: dict, rows: dict, arch: str, lr: float, seed: int) -> dict:
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bat = Batcher(pn, arch, rows["fit"], rng)
    net = make_net(arch, len(STEP_FEATURES), len(STATIC_FEATURES), pn["M"].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    vrows = rows["val"] if len(rows["val"]) <= VAL_N else np.sort(np.random.default_rng(12345).choice(rows["val"], VAL_N, replace=False))
    vb = [bat(vrows[i:i + 4096]) for i in range(0, len(vrows), 4096)]
    best, best_state, bad, curve = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(MAX_EP):
        net.train()
        er = rng.choice(rows["fit"], EPOCH_N, replace=False)
        tl = 0.0
        for i in range(0, EPOCH_N, BATCH):
            loss, _, _ = loss_fn(net, bat(er[i:i + BATCH]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(er[i:i + BATCH])
        net.eval()
        with torch.no_grad():
            vl = sum(float(loss_fn(net, b)[0]) * len(b[0]) for b in vb) / len(vrows)
        curve.append((tl / EPOCH_N, vl))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return {"net": net, "bat": bat, "val_loss": best, "curve": curve, "epochs": len(curve), "secs": time.time() - t0,
            "params": sum(p.numel() for p in net.parameters())}


def predict(net, bat: Batcher, rows: np.ndarray) -> np.ndarray:
    """(n, 3 horizons, 4) = mu, scale, df, logF."""
    torch = _torch()
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 4096):
            x, s, _, _ = bat(rows[i:i + 4096])
            out.append(torch.stack(net(x, s), dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


def cmd_train(arch: str, refit: str) -> None:
    torch = _torch()
    pn = panel()
    rows = split_rows(pn, refit)
    md, pd_ = root() / "models", root() / "preds"
    md.mkdir(exist_ok=True)
    pd_.mkdir(exist_ok=True)
    log = []

    def save(res: dict, lr: float, seed: int) -> None:
        name = f"{arch}_{refit}_lr{lr:g}_s{seed}"
        torch.save(res["net"].state_dict(), md / f"{name}.pt")
        meta = {"arch": arch, "refit": refit, "lr": lr, "seed": seed, "val_loss": res["val_loss"], "epochs": res["epochs"],
                "secs": res["secs"], "params": res["params"], "curve": res["curve"], "standardiser": res["bat"].state(),
                "step_features": STEP_FEATURES, "static_features": STATIC_FEATURES, "L": L, "d": D, "horizons": HS,
                "n_fit": int(len(rows["fit"])), "n_val": int(len(rows["val"])), "refit_dates": [d.isoformat() for d in REFITS[refit]]}
        (md / f"{name}.json").write_text(json.dumps(meta, default=float), encoding="utf-8")
        np.save(pd_ / f"{name}.npy", predict(res["net"], res["bat"], rows["test"]))
        log.append({k: meta[k] for k in ("arch", "refit", "lr", "seed", "val_loss", "epochs", "secs", "params")})
        print(json.dumps(log[-1]), flush=True)

    first = {}
    for lr in LRS:
        res = train_one(pn, rows, arch, lr, 0)
        first[lr] = res["val_loss"]
        save(res, lr, 0)
    lr_best = min(first, key=first.get)
    for s in SEEDS[1:]:
        save(train_one(pn, rows, arch, lr_best, s), lr_best, s)
    (root() / "logs").mkdir(exist_ok=True)
    (root() / "logs" / f"train_{arch}_{refit}.json").write_text(json.dumps({"chosen_lr": lr_best, "runs": log}, default=float), encoding="utf-8")
    np.save(root() / "preds" / f"rows_{refit}.npy", rows["test"])


# ------------------------------------------------------------------ classical


def t_fit(z: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    from scipy import stats

    z = z[np.isfinite(z)]
    if len(z) > 30000:
        z = np.random.default_rng(seed).choice(z, 30000, replace=False)
    df, loc, sc = stats.t.fit(z)
    return float(max(df, 2.05)), float(loc), float(sc)


def cmd_classical() -> None:
    pn = panel()
    out = {}
    for refit in REFITS:
        rows = split_rows(pn, refit)
        te = rows["test"]
        res = {k: np.full((len(te), 3, 4), np.nan, np.float32) for k in ("har", "base")}
        emp_q = np.full((len(te), 3, len(TAU99)), np.nan, np.float32)
        emp_F = np.full((len(te), 3), np.nan, np.float32)
        M, S = pn["M"], pn["S"]
        for a in range(len(SYMBOLS)):
            tr = rows["train"][pn["A"][rows["train"]] == a]
            ti = np.flatnonzero(pn["A"][te] == a)
            tei = te[ti]
            for j, h in enumerate(HS):
                lf_tr = M[tr, 1] if h == 1 else S[tr, STATIC_FEATURES.index(f"lfac_n{h}")]
                lf_te = M[tei, 1] if h == 1 else S[tei, STATIC_FEATURES.index(f"lfac_n{h}")]
                y, v = pn["Y"][tr, j].astype(float), pn["V"][tr, j].astype(float)
                # base-t
                df_, loc, sc = t_fit(y / np.exp(0.5 * lf_tr))
                res["base"][ti, j] = np.c_[np.full(len(ti), loc) * np.exp(0.5 * lf_te), sc * np.exp(0.5 * lf_te), np.full(len(ti), df_), lf_te]
                # log-HAR-t
                X = lambda rr, lf: np.c_[np.ones(len(rr)), M[rr, 0], lf, M[rr, 14], M[rr, 15], M[rr, 16]]  # noqa: E731
                b, *_ = np.linalg.lstsq(X(tr, lf_tr), np.log(np.maximum(v, 1e-6)), rcond=None)
                from ai_trader.ml.models import bias_const

                xb_tr = X(tr, lf_tr) @ b
                c = bias_const(v, xb_tr)
                lF_tr, lF_te = xb_tr + c, X(tei, lf_te) @ b + c
                df_, loc, sc = t_fit(y / np.exp(0.5 * lF_tr))
                res["har"][ti, j] = np.c_[loc * np.exp(0.5 * lF_te), sc * np.exp(0.5 * lF_te), np.full(len(ti), df_), lF_te]
            # empirical: trailing non-overlapping origins of the full series (labels ended by t)
            arow = np.flatnonzero(pn["A"] == a)
            for j, h in enumerate(HS):
                Ta = pn["T"][arow]
                hr = pn["hour"][arow]
                sel = (hr % h == 0) if h < 24 else (hr == 0)
                yy, vv = pn["Y"][arow, j].astype(float), pn["V"][arow, j].astype(float)
                cand = arow[sel & np.isfinite(yy) & np.isfinite(vv)]
                Tc = pn["T"][cand]
                for k, i in enumerate(tei):
                    t = pn["T"][i]
                    hi = np.searchsorted(Tc, t - np.timedelta64(h, "h"), side="right")
                    lo = max(0, hi - EMP_N[h])
                    if hi - lo < EMP_N[h] // 2:
                        continue
                    yw = pn["Y"][cand[lo:hi], j]
                    emp_q[ti[k], j] = np.quantile(yw, TAU99)
                    emp_F[ti[k], j] = pn["V"][cand[lo:hi], j].mean()
                del Ta
        np.save(root() / "preds" / f"har_{refit}.npy", res["har"])
        np.save(root() / "preds" / f"base_{refit}.npy", res["base"])
        np.save(root() / "preds" / f"emp_q_{refit}.npy", emp_q)
        np.save(root() / "preds" / f"emp_F_{refit}.npy", emp_F)
        out[refit] = int(len(te))
        print(refit, len(te), flush=True)


# ------------------------------------------------------------------ scoring


def t_quantiles(mu, sc, df, taus) -> np.ndarray:
    from scipy import stats

    return mu[:, None] + sc[:, None] * stats.t.ppf(np.asarray(taus)[None, :], df[:, None])


def pinball(q: np.ndarray, y: np.ndarray, taus) -> np.ndarray:
    tt = np.asarray(taus)[None, :]
    e = y[:, None] - q
    return np.mean(np.maximum(tt * e, (tt - 1) * e), axis=1)


def t_logpdf(y, mu, sc, df) -> np.ndarray:
    from scipy import stats

    return stats.t.logpdf(y, df, loc=mu, scale=sc)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    tr = sub.add_parser("train")
    tr.add_argument("--arch", required=True)
    tr.add_argument("--refit", required=True)
    sub.add_parser("classical")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build()
    elif a.cmd == "train":
        cmd_train(a.arch, a.refit)
    elif a.cmd == "classical":
        cmd_classical()
    else:
        from ai_trader.research.ws12_eval import evaluate

        evaluate(record=a.record)


if __name__ == "__main__":
    main()
