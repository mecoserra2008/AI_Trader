"""Plain-language 'distribution card' for one BTC state: what the field expects over 1h / 4h / 1d.

A card holds, per horizon: quantiles of the log return (in %), P(touch +x% / -x% before the horizon), P(at least one
jump), the typical move (sd), and which polynomial terms drive the diffusion and the jump intensity at this state
(contribution c_k * phi_k(X~) to ln D or ln p, largest first). No direction call is made: the drift is shown but
H-FLD2 registers it as a null.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ai_trader.field.model import Field, coords_matrix
from ai_trader.field.poly import design, mono_name
from ai_trader.field.propagate import origins_from_state, quantiles, simulate

QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
COORD_WORDS = {"x1": "short-term vol shock", "x2": "medium vs long vol", "L": "long-run vol level", "s": "hour-of-week",
               "jx": "recent jumps", "m24": "24h momentum", "fz": "funding z", "oi": "OI change", "dir": "trend score"}


def drivers(poly, Xs_row: np.ndarray, top: int = 4) -> list[dict]:
    T = design(Xs_row[None, :], poly.monos)[0]
    contrib = poly.coef * T
    out = []
    for k in np.argsort(-np.abs(contrib)):
        m = poly.monos[k]
        if sum(m) == 0:
            continue
        out.append({"term": mono_name(m, poly.names), "contribution": float(contrib[k]), "coef": float(poly.coef[k])})
        if len(out) >= top:
            break
    return out


def card(fld: Field, state: pl.DataFrame, i: int, *, horizons=(1, 4, 24), moves=(0.01, 0.02, 0.03, 0.05), n_paths: int = 4000, seed: int = 0) -> dict:
    row = state.slice(i, 1)
    X = coords_matrix(row)
    Xs = fld.Xs(X)[0]
    lnvf = float(np.log(row["vf"][0]))
    mu1, D1, p1 = fld.params(X, np.array([lnvf]))
    levels = tuple([+m for m in moves] + [-m for m in moves])
    levels_log = tuple(math.log(1 + m) for m in levels)
    out = {"as_of": str(row["known_ts"][0]), "close": float(row["close"][0]), "cell": row["cell"][0] if "cell" in row.columns else None,
           "state": {c: float(v) for c, v in zip(fld.std.names, X[0], strict=True)},
           "one_hour": {"sd_pct": 100 * math.sqrt(float(D1[0]) + float(p1[0]) * float(D1[0]) * fld.kou.second), "p_jump": float(p1[0]),
                        "drift_pct": 100 * float(mu1[0])},
           "drivers_D": drivers(fld.logD, Xs), "drivers_lam": drivers(fld.loglam, Xs), "horizons": {}}
    for h in horizons:
        org = origins_from_state(state, [i], fld.profile, h)
        sim = simulate(fld, org, h, n_paths, seed=seed, levels=levels_log)
        term = sim["terminal"][0]
        q = quantiles(term[None, :], QS)[0]
        out["horizons"][f"{h}h"] = {
            "quantiles_pct": {f"q{int(100 * a):02d}": 100 * (math.exp(v) - 1) for a, v in zip(QS, q, strict=True)},
            "sd_pct": 100 * float(np.std(term)),
            "p_jump": float(sim["p_jump"][0]),
            "p_touch_pct": {f"{100 * lv:+.0f}%": float(sim["p_hit"][lb][0]) for lv, lb in zip(levels, levels_log, strict=True)},
            "p_up_close": float(np.mean(term > 0)),
        }
    out["text"] = card_text(out)
    return out


def _words(drv: list[dict]) -> str:
    bits = []
    for d in drv[:3]:
        nm = d["term"]
        for k, v in COORD_WORDS.items():
            nm = nm.replace(k, v) if nm == k else nm
        bits.append(f"{nm} ({'raises' if d['contribution'] > 0 else 'lowers'} it {math.exp(abs(d['contribution'])):.2f}x)")
    return ", ".join(bits) if bits else "nothing beyond the base vol"


def card_text(c: dict) -> str:
    L = [f"BTC at {c['close']:.0f} ({c['as_of']}, regime {c['cell']}).",
         f"Next hour: typical move {c['one_hour']['sd_pct']:.2f}%, jump probability {100 * c['one_hour']['p_jump']:.1f}%."]
    for h, v in c["horizons"].items():
        q = v["quantiles_pct"]
        t = v["p_touch_pct"]
        L.append(f"{h}: 90% of outcomes between {q['q05']:+.2f}% and {q['q95']:+.2f}% (median {q['q50']:+.2f}%); "
                 f"touch +1% {100 * t['+1%']:.0f}%, -1% {100 * t['-1%']:.0f}%, +2% {100 * t['+2%']:.0f}%, -2% {100 * t['-2%']:.0f}%, "
                 f"+3% {100 * t['+3%']:.0f}%, -3% {100 * t['-3%']:.0f}%; P(jump) {100 * v['p_jump']:.0f}%.")
    L.append(f"Diffusion driven by: {_words(c['drivers_D'])}.")
    L.append(f"Jump intensity driven by: {_words(c['drivers_lam'])}.")
    L.append("Direction: no skill expected (H-FLD2 null); read the quantiles as a symmetric range, not a call.")
    return "\n".join(L)


__all__ = ["QS", "card", "card_text", "drivers"]
