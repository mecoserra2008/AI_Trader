"""WS08 runner: build panels, then each analysis stage; outputs JSON under data/research/ws08/.

    .venv/Scripts/python.exe scripts/ws08_run.py build|grid|qr|econ|xsect|macro|rv|size
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time

import numpy as np

from ai_trader.research import ws08_predict as w

OUT = w.OUT
PANEL = OUT / "panel.pkl"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not (isinstance(v, np.ndarray) and v.size > 50)}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    return o


def dump(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(_clean(obj), indent=1), encoding="utf-8")
    print(f"wrote {name}.json", flush=True)


def load_panel():
    with PANEL.open("rb") as f:
        return pickle.load(f)


def main(stage: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if stage == "build":
        P = w.build()
        with PANEL.open("wb") as f:
            pickle.dump(P, f)
        # coverage summary
        cov = {}
        for fr in ("hourly", "daily"):
            g = P[fr]["grid"]
            for a, d in P[fr]["x"].items():
                for k, v in d.items():
                    fin = np.isfinite(v)
                    if fin.any():
                        cov[f"{fr}:{a}:{k}"] = [int(fin.sum()), int(g[fin][0]), int(g[fin][-1])]
        dump("coverage", cov)
    elif stage == "grid":
        P = load_panel()
        G = w.analyse_grid(P, log=print)
        dump("grid", G)
    elif stage == "econ":
        P = load_panel()
        G = json.loads((OUT / "grid.json").read_text())
        cells = [c for c in G["cells"] if "b" in c]
        res = {}
        for c in cells:
            try:
                res[c["cell"]] = w.sign_rule(P, c)
            except Exception as e:  # noqa: BLE001
                res[c["cell"]] = {"error": str(e)}
        srs = np.array([v["is"]["sharpe_per_trade"] for v in res.values() if "is" in v and "sharpe_per_trade" in v["is"]])
        for k, v in res.items():
            if "is" in v and "sharpe_per_trade" in v["is"]:
                i = v["is"]
                v["is"]["dsr"] = w.deflated_sharpe(i["sharpe_per_trade"], i["n_trades"], i["skew"], i["kurt"], srs)
        dump("econ", {"n_trials": len(srs), "sr_var": float(np.var(srs)), "cells": res})
    elif stage == "qr":
        P = load_panel()
        dump("qr", w.analyse_qr(P, log=print))
    elif stage == "xsect":
        from ai_trader.core.time import now

        X = w.xsect_panel(now(), log=print)
        CH = w.xsect_chars(X)
        XS = w.analyse_xsect(X, CH, log=print)
        TS = w.analyse_trend_sleeve(X, CH, log=print)
        dump("xsect", {"names": X["names"], "last_bar": X["last_bar"], "xs": XS, "trend_sleeve": TS})
    elif stage == "macro":
        P = load_panel()
        dump("macro", w.analyse_macro(P, log=print))
    elif stage == "rv":
        P = load_panel()
        dump("rv", w.analyse_rv(P, log=print))
    elif stage == "size":
        rng = np.random.default_rng(11)
        out = {}
        for k in (1, 7, 30):
            rej, rej_o = 0, 0
            for _ in range(300):
                u = rng.standard_normal(2001)
                e = -0.9 * u + math.sqrt(1 - 0.81) * rng.standard_normal(2001)
                x = np.zeros(2001)
                for t in range(1, 2001):
                    x[t] = 0.99 * x[t - 1] + u[t]
                z1, xx = e[1:], x[:-1]
                rej += w.ivx_wald(z1, xx, k=k)["p"] < 0.05
                y = w.forward_sum_valid(np.r_[np.nan, z1], k)[:-1] if k > 1 else z1
                m = np.isfinite(y)
                rej_o += abs(w.ols_nw(y[m], xx[m], w.nw_lag(int(m.sum()), k))["t"]) > 1.96
            out[f"k{k}"] = {"ivx_size": rej / 300, "nw_t_size": rej_o / 300}
            print(k, out[f"k{k}"], flush=True)
        dump("size", {"design": "T=2000, rho=0.99, corr(e,u)=-0.9, beta=0, 300 reps", **out})
    print(f"{stage} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
