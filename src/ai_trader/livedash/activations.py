"""Forward pass of the frozen PNNs on one journal line, decomposed per edge (read only; torch on 1 thread).

For a stored line (data/journal/live_nn/<date>.jsonl, key `inputs`) and each horizon (1h net: wf_1h, 24h: wf_1d):
  inputs     raw values as stored, standardised with the checkpoint's fit-block Standardizer, clipped to +-6
             (ai_trader.live.nn.Net._Z; a missing input enters as 0 and the stored quantiles are null, as in the live run)
  layer 1    encoder KANLayer n_in -> 4: edge outputs phi_ji(x_i) (pnn.KANLayer.edges) and node values z_j = b_j + sum_i phi_ji
  layer 2    head KANLayer 4 -> 7: edge outputs phi_kj(z_j) and raw head outputs o_k
  quantiles  pnn.quantiles_from_raw per seed; the 2-seed mean in model units (y = r / sigma_ref) x 100 sigma_ref -> %;
             computed with the model's own forward (the same calls as Net.quantiles), so they equal the stored line
  latent     seed-0 z (what the line stores) and zhat_next = LatentDynamics.predict(z, u(S + 1h)) (1h net only)
  scale      sigma_ref = exp(lsig2 / 2) (1h) or exp(lrv_w / 2) (24h); the stored sigma_ref_pct is used when present
Attribution per input i (sigma = (q90 - q10) / 2.5631 in %, scale held fixed, 2-seed mean network):
  l1_abs                 sum_j |phi_ji(x_i)| over the encoder edges out of input i (mean over seeds)
  mean_ablation_sigma    sigma(now) - sigma(x_i at its training mean, i.e. standardised 0): the input's effect on sigma
  week_ablation_dln      ln sigma(now) - ln sigma(x_i at its trailing-week mean over the reference lines), one at a time
                         (the 'drivers vs the weekly mean'); the scale's own share is ln(scale now) - mean ln(scale week)
Edge catalogue (`edge_catalogue`, per net, static): every edge function's exact form
  phi(x) = w silu(x) + sum_d c_d T_d(u), u = tanh(x / 2), rewritten as a polynomial in u, a cubic least-squares fit in x
  over a grid (pnn.poly_fit / poly_str) with its R^2, and the sampled curve for the page's tooltip.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

NORM80 = 2.5631031310892007
NET_H = {"1h": "1h", "24h": "1d"}
SCALE_INPUT = {"1h": "lsig2", "24h": "lrv_w"}
GRID_N = 33


def _torch():
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch


def _utc(t) -> datetime:
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


@lru_cache(maxsize=4)
def net(h: str, pnn_dir: str):
    """(Net, [seed models]) for live horizon h ('1h' | '24h'); the frozen checkpoint via ai_trader.live.nn.load_net."""
    _torch()
    from ai_trader.live import nn

    n = nn.load_net(NET_H[h], pnn_dir)
    return n, [f.model() for f in n.fits]


def calendar_u(t: datetime) -> np.ndarray:
    """The latent dynamics' control input for step t (the same as live.nn: hour and weekday on the unit circle)."""
    return np.array([math.sin(2 * math.pi * t.hour / 24), math.cos(2 * math.pi * t.hour / 24),
                     math.sin(2 * math.pi * (t.isoweekday() - 1) / 7), math.cos(2 * math.pi * (t.isoweekday() - 1) / 7)])


def _raw(line: dict, h: str, feats) -> np.ndarray:
    d = line["inputs"][h]
    return np.array([np.nan if d.get(c) is None else float(d[c]) for c in feats], float)


def _scale(line: dict, h: str, raw: np.ndarray, feats) -> float:
    s = line.get("net", {}).get(h, {}).get("sigma_ref_pct")
    if s is not None:
        return float(s) / 100
    v = raw[list(feats).index(SCALE_INPUT[h])]
    return math.exp(0.5 * v) if np.isfinite(v) else float("nan")


def _ens(models, X) -> np.ndarray:
    """2-seed mean quantiles in model units, exactly as QFit.predict + pnn.ensemble_quantiles."""
    torch = _torch()
    x = torch.as_tensor(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        qs = [m(x, None).numpy().astype(float) for m in models]
    return np.mean(np.stack(qs, axis=0), axis=0)


def _sigma_model(q: np.ndarray) -> np.ndarray:
    return (q[..., 5] - q[..., 1]) / NORM80


def forward(line: dict, h: str, pnn_dir: str, week_lines: list[dict] | None = None) -> dict:
    torch = _torch()
    from ai_trader.ml import pnn

    n, models = net(h, pnn_dir)
    feats = list(n.features)
    raw = _raw(line, h, feats)
    missing = ~np.isfinite(raw)
    zraw = n.zraw(raw[None, :])[0]
    Z = n._Z(raw[None, :])                       # (1, n_in), float64, NaN -> 0, clipped
    scale = _scale(line, h, raw, feats)
    x = torch.as_tensor(np.asarray(Z, dtype=np.float32))
    seeds, edge_err = [], 0.0
    _, _, _, q_from_raw = pnn.classes()
    with torch.no_grad():
        for m in models:
            E1 = m.enc.edges(x)                  # (1, k, n_in)
            z = m.enc(x)                         # (1, k)
            E2 = m.head.edges(z)                 # (1, 7, k)
            o = m.head(z)                        # (1, 7)
            qm = q_from_raw(o)
            edge_err = max(edge_err, float((E1.sum(-1) + m.enc.bias - z).abs().max()), float((E2.sum(-1) + m.head.bias - o).abs().max()))
            seeds.append({"latent": z[0].tolist(), "latent_bias": m.enc.bias.tolist(), "enc_edges": E1[0].tolist(),
                          "head_edges": E2[0].tolist(), "head_raw": o[0].tolist(), "head_bias": m.head.bias.tolist(),
                          "quantiles_model": qm[0].tolist()})
    q = _ens(models, Z)[0]
    if missing.any():
        q = np.full_like(q, np.nan)
    qpct = [float(100 * v * scale) for v in q]   # the same expression as live.nn._qdict
    stored = line.get("net", {}).get(h, {}).get("quantiles_pct") or {}
    st = [stored.get(k) for k in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")]
    diff = max((abs(a - b) for a, b in zip(qpct, st, strict=True) if a is not None and b is not None and math.isfinite(a)), default=None)
    sig_now = float(_sigma_model(q) * 100 * scale)

    # ---- attribution
    k = len(feats)
    l1 = np.mean([np.abs(np.asarray(s["enc_edges"])).sum(axis=0) for s in seeds], axis=0)
    Za = np.repeat(Z, k, axis=0)
    Za[np.arange(k), np.arange(k)] = 0.0
    qa = _ens(models, Za)
    abl_sig = sig_now - _sigma_model(qa) * 100 * scale
    abl_q50 = 100 * scale * (q[3] - qa[:, 3])
    att = {"l1_abs": l1.tolist(), "mean_ablation_sigma_pct": abl_sig.tolist(), "mean_ablation_q50_pct": abl_q50.tolist()}
    wk = week_reference(week_lines or [], h, feats)
    if wk["n"]:
        Zw = np.repeat(Z, k, axis=0)
        zw = np.clip(np.nan_to_num(n.zraw(wk["mean_raw"][None, :])[0], nan=0.0), -pnn.CLIP, pnn.CLIP)
        Zw[np.arange(k), np.arange(k)] = zw
        qw = _ens(models, Zw)
        sm = _sigma_model(q)
        att["week_ablation_dln_sigma"] = (np.log(sm) - np.log(_sigma_model(qw))).tolist()
        att["week"] = {"n": wk["n"], "from": wk["from"], "to": wk["to"], "mean_raw": wk["mean_raw"].tolist(),
                       "mean_z": zw.tolist(), "mean_sigma_pct": wk["mean_sigma_pct"],
                       "dln_sigma": float(math.log(sig_now) - wk["mean_ln_sigma"]) if sig_now > 0 else None,
                       "dln_scale": float(math.log(scale) - wk["mean_ln_scale"]),
                       "dln_model": float(math.log(sm) - wk["mean_ln_model"])}
    else:
        att["week_ablation_dln_sigma"] = None
        att["week"] = {"n": 0}

    out = {"h": h, "net": NET_H[h], "refit_month": n.month, "pt_sha256": n.sha_pt, "features": feats,
           "raw": [None if not np.isfinite(v) else float(v) for v in raw], "zraw": [None if not np.isfinite(v) else float(v) for v in zraw],
           "z": Z[0].tolist(), "missing": missing.tolist(), "scale_pct": 100 * scale, "quantiles_model": q.tolist(), "quantiles_pct": qpct,
           "stored_quantiles_pct": st, "max_abs_diff_vs_stored_pct": diff, "sigma_pct": sig_now, "seeds": seeds,
           "attribution": att, "check": {"edge_sum_max_err": edge_err}}
    z0 = np.asarray(seeds[0]["latent"], float)
    lat = {"z": z0.tolist(), "zmu": n.dyn.zmu.tolist(), "zsd": n.dyn.zsd.tolist()}
    if h == "1h":
        S = _utc(line["forecast_hour"])
        lat["zhat_next"] = n.dyn.predict(z0[None, :], calendar_u(S + timedelta(hours=1))[None, :])[0].tolist()
        lat["stored_z"] = line.get("latent", {}).get("z")
        lat["stored_zhat_next"] = line.get("latent", {}).get("zhat_next")
    else:
        lat["stored_z"] = line.get("latent", {}).get("z_24h_net")
    out["latent"] = lat
    return out


def week_reference(lines: list[dict], h: str, feats) -> dict:
    """Means over the reference lines (normally the 168 h up to and including the line): raw inputs, ln sigma,
    ln scale and ln sigma in model units (sigma_pct / sigma_ref_pct)."""
    X, ls, lsc, lm, ts = [], [], [], [], []
    for ln in lines:
        nh = ln.get("net", {}).get(h, {})
        try:
            x = np.array([float(ln["inputs"][h][c]) for c in feats], float)
            s, r = float(nh["sigma_pct"]), float(nh["sigma_ref_pct"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(x).all() or not (s > 0 and r > 0):
            continue
        X.append(x)
        ls.append(math.log(s))
        lsc.append(math.log(r / 100))
        lm.append(math.log(s / r))
        ts.append(ln["forecast_hour"])
    if not X:
        return {"n": 0}
    return {"n": len(X), "from": min(ts), "to": max(ts), "mean_raw": np.mean(X, axis=0), "mean_ln_sigma": float(np.mean(ls)),
            "mean_sigma_pct": float(np.mean(np.exp(ls))), "mean_ln_scale": float(np.mean(lsc)), "mean_ln_model": float(np.mean(lm))}


def cheb_to_mono(c) -> list[float]:
    """sum_{d=1..4} c_d T_d(u) as monomial coefficients a_0..a_4 in u (T1 = u, T2 = 2u^2 - 1, T3 = 4u^3 - 3u,
    T4 = 8u^4 - 8u^2 + 1). General degree via the Chebyshev recursion."""
    D = len(c)
    T = [[1.0], [0.0, 1.0]]
    for _ in range(2, D + 1):
        a, b = T[-1], T[-2]
        nxt = [0.0] + [2 * v for v in a]
        for i, v in enumerate(b):
            nxt[i] -= v
        T.append(nxt)
    out = [0.0] * (D + 1)
    for d in range(1, D + 1):
        for i, v in enumerate(T[d]):
            out[i] += float(c[d - 1]) * v
    return out


@lru_cache(maxsize=4)
def edge_catalogue(h: str, pnn_dir: str) -> dict:
    """Static description of every edge function of both seeds (exact form, cubic fit, sampled curve)."""
    torch = _torch()
    from ai_trader.ml import pnn

    n, models = net(h, pnn_dir)
    feats = list(n.features)
    zmu, zsd = np.asarray(n.dyn.zmu, float), np.asarray(n.dyn.zsd, float)
    gx = np.linspace(-4, 4, GRID_N)
    lat_names = [f"z{j + 1}" for j in range(len(zmu))]
    out = {"h": h, "features": feats, "latent_names": lat_names, "grid_enc": gx.tolist(), "seeds": []}
    for m in models:
        layers = {}
        for name, layer, ins, grid in (("enc", m.enc, feats, np.repeat(gx[:, None], len(feats), axis=1)),
                                       ("head", m.head, lat_names, zmu[None, :] + np.linspace(-3.5, 3.5, GRID_N)[:, None] * zsd[None, :])):
            with torch.no_grad():
                E = layer.edges(torch.as_tensor(grid, dtype=torch.float32)).numpy().astype(float)  # (G, out, in)
            coef, wres = layer.coef.detach().numpy().astype(float), layer.w_res.detach().numpy().astype(float)
            rows = []
            for j in range(E.shape[1]):
                row = []
                for i in range(E.shape[2]):
                    f = E[:, j, i]
                    fit = pnn.poly_fit(grid[:, i], f, 3)[-1]
                    var = ins[i]
                    mono = cheb_to_mono(coef[j, i])
                    exact = f"{wres[j, i]:+.3f} silu({var}) " + ("+ " if not pnn.poly_str(mono, "u").startswith("-") else "") + pnn.poly_str(mono, "u") \
                        + f", u = tanh({var}/2)"
                    row.append({"w_res": float(wres[j, i]), "cheb": coef[j, i].tolist(), "mono_u": mono, "exact": exact,
                                "cubic": fit["coef"], "cubic_r2": fit["r2"], "cubic_str": pnn.poly_str(fit["coef"], var),
                                "curve": [round(float(v), 5) for v in f], "importance_grid": float(np.std(f))})
                rows.append(row)
            layers[name] = {"edges": rows, "grid": grid[:, 0].tolist() if name == "enc" else grid.T.tolist(), "bias": layer.bias.detach().numpy().astype(float).tolist()}
        out["seeds"].append(layers)
    return out


def activations(line: dict, pnn_dir: str | Path, week_lines: list[dict] | None = None) -> dict:
    """Both nets for one line. Raises KeyError if the line has no stored inputs."""
    pd = str(pnn_dir)
    out = {"id": line["id"], "forecast_hour": line["forecast_hour"], "horizons": {}}
    for h in ("1h", "24h"):
        if h in line.get("inputs", {}):
            out["horizons"][h] = forward(line, h, pd, week_lines)
    return out


__all__ = ["activations", "calendar_u", "cheb_to_mono", "edge_catalogue", "forward", "net", "week_reference"]
