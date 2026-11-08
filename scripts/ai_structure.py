"""AI trade structure breaks vs history, BTC linkage, and crypto-AI betas. Descriptive (no trading claim)."""
import json, time, urllib.request, math
from datetime import datetime, timezone, timedelta
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(2)


def yahoo(sym, rng="15y"):
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"
    d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30))["chart"]["result"][0]
    ts = d["timestamp"]; q = d["indicators"]
    c = (q.get("adjclose") or [{}])[0].get("adjclose") or q["quote"][0]["close"]
    df = pl.DataFrame({"d": [datetime.fromtimestamp(x, timezone.utc).date() for x in ts], sym: c}).drop_nulls()
    return df.unique("d", keep="last").sort("d")


def bybit_daily(sym):
    out, end = [], int(time.time() * 1000)
    for _ in range(3):
        u = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={sym}&interval=D&limit=1000&end={end}"
        r = json.load(urllib.request.urlopen(u, timeout=30))["result"]["list"]
        if not r: break
        out += r; end = int(r[-1][0]) - 1
        if len(r) < 1000: break
    df = pl.DataFrame({"d": [datetime.fromtimestamp(int(x[0]) / 1000, timezone.utc).date() for x in out], sym: [float(x[4]) for x in out]})
    return df.unique("d").sort("d")


eq = {s: yahoo(s) for s in ["NVDA", "SMH", "^NDX", "BTC-USD", "IREN", "CORZ", "WULF", "CRWV", "MSTR", "COIN", "CIFR", "HUT"]}
cr = {s: bybit_daily(s) for s in ["TAOUSDT", "RENDERUSDT", "FETUSDT", "VIRTUALUSDT", "NEARUSDT", "ETHUSDT"]}

# ---------------------------------------------------------------- 1. weekly structure
def weekly(df, col):
    return df.with_columns(pl.col("d").cast(pl.Datetime).dt.truncate("1w").alias("w")).group_by("w").agg(pl.col(col).last(), pl.col("d").last()).sort("w")

def structure_events(sym, k=2, min_up=0.15):
    w = weekly(eq[sym], sym)
    c = w[sym].to_numpy(); dates = w["d"].to_list(); n = len(c)
    # confirmed pivots: swing low at i known at i+k
    lows = [i for i in range(k, n - k) if c[i] == min(c[i - k:i + k + 1])]
    highs = [i for i in range(k, n - k) if c[i] == max(c[i - k:i + k + 1])]
    ev = []
    last_break = -999
    for t in range(k + 5, n):
        known_lows = [i for i in lows if i + k <= t - 1]
        if not known_lows: continue
        sl = known_lows[-1]
        peak_i = int(np.argmax(c[max(0, t - 52):t])) + max(0, t - 52)
        run_up = c[peak_i] / c[max(0, peak_i - 52):peak_i + 1].min() - 1
        # bearish BOS: weekly close below the latest confirmed swing low, after an uptrend (>= min_up off the 1y low), first break in 13 weeks
        if c[t] < c[sl] and c[t - 1] >= c[sl] and run_up >= min_up and t - last_break >= 13:
            last_break = t
            dd = c[t] / c[peak_i] - 1
            fwd = {h: (c[t + h] / c[t] - 1) if t + h < n else np.nan for h in (4, 13, 26, 52)}
            fmin = c[t:min(n, t + 27)].min() / c[t] - 1
            new_high = next((j - t for j in range(t + 1, min(n, t + 79)) if c[j] > c[peak_i]), None)
            ev.append({"sym": sym, "break_week": dates[t], "dd_at_break": dd, "fwd4w": fwd[4], "fwd13w": fwd[13], "fwd26w": fwd[26], "fwd52w": fwd[52],
                       "further_dd_26w": fmin, "weeks_to_new_high": new_high})
    return pl.DataFrame(ev), w, lows, highs

print("## 1. Weekly bearish breaks of structure (close below last confirmed swing low after a >=15% run-up)\n")
allev = []
for s in ["NVDA", "SMH", "^NDX"]:
    ev, w, lows, highs = structure_events(s)
    allev.append(ev)
    c = w[s].to_numpy()
    peak = c[-52:].max(); dd = c[-1] / peak - 1
    ma40 = c[-40:].mean()
    print(f"{s}: last weekly close {c[-1]:.2f} ({w['d'][-1]}), {100*dd:.1f}% from 52w high, {100*(c[-1]/ma40-1):+.1f}% vs 40w MA (~200d)")
    print(ev.with_columns([pl.col(x) * 100 for x in ["dd_at_break", "fwd4w", "fwd13w", "fwd26w", "fwd52w", "further_dd_26w"]]).tail(14))
    print()

ev = pl.concat(allev)
hist = ev.filter(pl.col("break_week") < datetime(2026, 10, 1).date())
print("Summary over historical breaks (excl. the current one):")
print(hist.group_by("sym").agg(pl.len().alias("n"), (pl.col("fwd13w").median() * 100).alias("fwd13w_med"), ((pl.col("fwd13w") > 0).mean()).alias("p13w_up"),
      (pl.col("further_dd_26w").median() * 100).alias("further_dd_med"), pl.col("weeks_to_new_high").median().alias("wk_to_new_high_med"),
      pl.col("weeks_to_new_high").is_null().mean().alias("share_no_new_high_18m")))

# BTC around NDX / SMH breaks
btc = eq["BTC-USD"]
def btc_fwd(d0, h):
    a = btc.filter(pl.col("d") >= d0).head(1); b = btc.filter(pl.col("d") >= d0 + timedelta(days=h)).head(1)
    return float(b["BTC-USD"][0] / a["BTC-USD"][0] - 1) if a.height and b.height else np.nan
rows = []
for r in ev.filter(pl.col("sym").is_in(["SMH", "^NDX"])).iter_rows(named=True):
    if r["break_week"] < datetime(2018, 1, 1).date(): continue
    pre = btc.filter((pl.col("d") <= r["break_week"]) & (pl.col("d") >= r["break_week"] - timedelta(days=60)))
    rows.append({"sym": r["sym"], "break_week": r["break_week"], "btc_prior60d": float(pre["BTC-USD"][-1] / pre["BTC-USD"][0] - 1) * 100 if pre.height > 2 else np.nan,
                 "btc_fwd4w": btc_fwd(r["break_week"], 28) * 100, "btc_fwd13w": btc_fwd(r["break_week"], 91) * 100, "btc_fwd26w": btc_fwd(r["break_week"], 182) * 100,
                 "ai_fwd13w": r["fwd13w"] * 100})
bt = pl.DataFrame(rows)
print("\n## 2. BTC after AI-index structure breaks (2018+)")
print(bt)
print("BTC fwd13w median %.1f%%, share up %.2f (unconditional 13w median %.1f%%)" % (bt["btc_fwd13w"].median(), (bt["btc_fwd13w"] > 0).mean(),
      100 * float(np.nanmedian(btc["BTC-USD"].to_numpy()[91:] / btc["BTC-USD"].to_numpy()[:-91] - 1))))

# ---------------------------------------------------------------- 3. rolling correlation BTC vs NDX / NVDA (weekly, 26w windows)
def wret(df, col):
    w = weekly(df, col)
    return w.select("w", (pl.col(col).log().diff()).alias(col))
W = wret(btc, "BTC-USD")
for s in ["^NDX", "NVDA", "SMH"]:
    W = W.join(wret(eq[s], s), on="w", how="inner")
W = W.drop_nulls().sort("w")
for s in ["^NDX", "NVDA"]:
    W = W.with_columns(pl.rolling_corr(pl.col("BTC-USD"), pl.col(s), window_size=26).alias(f"corr_{s}"))
yr = W.with_columns(pl.col("w").dt.year().alias("y")).group_by("y").agg(pl.col("corr_^NDX").mean(), pl.col("corr_NVDA").mean()).sort("y")
print("\n## 3. BTC vs NDX / NVDA, 26-week rolling correlation of weekly returns (yearly mean; last value)")
print(yr)
print("latest 26w corr: NDX %.2f, NVDA %.2f" % (W["corr_^NDX"][-1], W["corr_NVDA"][-1]))
# BTC beta to NDX over last 26w vs history
def beta(y, x): return float(np.cov(y, x)[0, 1] / np.var(x))
print("BTC beta to NDX: last 26w %.2f, 2021-2026 full %.2f" % (beta(W["BTC-USD"][-26:].to_numpy(), W["^NDX"][-26:].to_numpy()), beta(W.filter(pl.col("w") >= datetime(2021, 1, 1))["BTC-USD"].to_numpy(), W.filter(pl.col("w") >= datetime(2021, 1, 1))["^NDX"].to_numpy())))

# ---------------------------------------------------------------- 4. crypto-AI betas (weekly, last 52 weeks): r = a + b_btc BTC + b_nvda NVDA_resid
print("\n## 4. Weekly betas over the last 52 weeks: r = a + b_BTC*BTC + b_AI*(NVDA orthogonalised to BTC)")
base = wret(btc, "BTC-USD").join(wret(eq["NVDA"], "NVDA"), on="w").drop_nulls().sort("w").tail(52)
x1 = base["BTC-USD"].to_numpy(); x2 = base["NVDA"].to_numpy()
b = np.polyfit(x1, x2, 1); x2r = x2 - np.polyval(b, x1)
rows = []
names = {**{s: eq[s] for s in ["IREN", "CORZ", "WULF", "CIFR", "HUT", "CRWV", "MSTR", "COIN"]}, **cr}
for s, df in names.items():
    r = wret(df, s).join(base.select("w"), on="w", how="inner").sort("w")
    j = base.join(r, on="w", how="inner").drop_nulls()
    if j.height < 30: continue
    y = j[s].to_numpy(); X = np.column_stack([np.ones(j.height), j["BTC-USD"].to_numpy(), (j["NVDA"].to_numpy() - np.polyval(b, j["BTC-USD"].to_numpy()))])
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef; r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # share of explained variance from the AI factor
    r2_btc = 1 - ((y - np.polyval(np.polyfit(X[:, 1], y, 1), X[:, 1])) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    resid = y - yhat; se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
    rows.append({"asset": s.replace("USDT", ""), "n_w": j.height, "beta_BTC": coef[1], "t_BTC": coef[1] / se[1], "beta_AI": coef[2], "t_AI": coef[2] / se[2], "R2": r2, "R2_BTC_only": r2_btc,
                 "ret_since_Oct15_%": float(df.filter(pl.col("d") >= datetime(2026, 10, 15).date())[s][-1] / df.filter(pl.col("d") >= datetime(2026, 10, 15).date())[s][0] - 1) * 100})
print(pl.DataFrame(rows).sort("t_AI", descending=True))
for s in ["NVDA", "SMH", "^NDX", "BTC-USD"]:
    d = eq[s]; print(s, "since 2026-10-15: %.1f%%" % (100 * (d[s][-1] / d.filter(pl.col("d") >= datetime(2026, 10, 15).date())[s][0] - 1)), " last", d["d"][-1])
