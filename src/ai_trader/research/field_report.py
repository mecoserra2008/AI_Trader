"""Analysis and report for the BTC field study (H-FLD1-3): reads the monthly checkpoints of research.field_study, runs
H-FLD3, the descriptive vol field, the Fokker-Planck check, the delta check and the distribution cards, records the
verdicts (once per segment) and writes research/reports/btc_field_<date>.{json,md} plus data/models/field/."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.field.cards import card
from ai_trader.field.kramers_moyal import binned_moments, ou_correct
from ai_trader.field.model import Field, coords_matrix
from ai_trader.field.poly import (
    SparsePoly,
    design,
    glm_fit,
    mono_name,
    monomials,
    stlsq,
)
from ai_trader.field.propagate import (
    cdf_quantiles,
    fp2d_logprice_vol,
    mc_logprice_vol,
    origins_from_state,
    simulate,
)
from ai_trader.field.state import COORDS, build_state, flow_delta, load_frame
from ai_trader.research import field_study as fs
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t

QS = [0.05, 0.25, 0.5, 0.75, 0.95]


def _load_months() -> pl.DataFrame:
    files = sorted((fs.SCR / "months").glob("*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def _load_fits() -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted((fs.SCR / "fits").glob("*.json"))]


def _seg(ev: pl.DataFrame, seg: str) -> pl.DataFrame:
    cut = knowledge_cutoff()
    return ev.filter(pl.col("origin") < cut) if seg == "llm_in_sample" else ev.filter(pl.col("origin") >= cut)


# ------------------------------------------------------------------ descriptive vol field + FP check


def vol_field(fld: Field, st: pl.DataFrame, tr: np.ndarray) -> dict:
    """KM of the log-vol coordinate v = 0.5 ln vf: binned moments, OU-corrected linear drift, sparse polynomial drift /
    diffusion / return-vol correlation on a degree-2 library (STLSQ |t| >= 3)."""
    v = 0.5 * np.log(st["vf"].to_numpy())
    dv = np.r_[v[1:] - v[:-1], np.nan]
    r = st["r_next"].to_numpy().astype(float)
    X = coords_matrix(st)
    ok = tr & np.isfinite(dv) & np.isfinite(r)
    mu, D, p = fld.params(X[ok], np.log(st["vf"].to_numpy()[ok]))
    z = (r[ok] - mu) / np.sqrt(D)
    edges = np.quantile(v[ok], np.linspace(0, 1, 11))
    b = binned_moments(v[ok], np.column_stack([dv[ok], z]), edges, 1.0, 30)
    ou = ou_correct(v[ok], dv[ok], 1.0)
    ou_x1 = ou_correct(X[ok, 0], np.r_[X[1:, 0] - X[:-1, 0], np.nan][ok], 1.0)
    Xs = fld.Xs(X[ok])
    lib = monomials(len(COORDS), 2)
    T = design(Xs, lib)
    sup_m, res_m = stlsq(T, dv[ok], "gaussian", t_min=3.0)
    e = dv[ok] - T[:, sup_m] @ res_m.coef
    sup_d, res_d = stlsq(T, e * e, "gamma", t_min=3.0)
    sd_v = np.sqrt(np.exp(T[:, sup_d] @ res_d.coef))
    sup_c, res_c = stlsq(T, z * e / sd_v, "gaussian", t_min=3.0)
    mk = lambda sup, res, link, fam, off: SparsePoly.from_fit(COORDS, [lib[i] for i in sup], res, link, fam, off)
    pm, pdd, pc = mk(sup_m, res_m, "identity", "gaussian", ""), mk(sup_d, res_d, "log", "gamma", ""), mk(sup_c, res_c, "identity", "gaussian", "")
    return {
        "coordinate": "v = 0.5 ln vf (fast deseasonalised log vol, per hour)",
        "binned": {"v_center": b["center"].tolist(), "n": b["n"].tolist(), "drift_per_h": b["m1"].tolist(), "D_v_per_h": b["var"].tolist(),
                   "corr_r_dv": b["corr"].tolist()},
        "ou_v": {"theta_per_h": ou.theta, "half_life_h": ou.half_life, "mean": ou.mean, "D": ou.D, "theta_naive": ou.theta_naive},
        "ou_x1": {"theta_per_h": ou_x1.theta, "half_life_h": ou_x1.half_life, "mean": ou_x1.mean, "D": ou_x1.D, "theta_naive": ou_x1.theta_naive},
        "mu_v": pm.to_dict() | {"equation": pm.equation("E[dv]")}, "D_v": pdd.to_dict() | {"equation": pdd.equation("D_v")},
        "rho_rv": pc.to_dict() | {"equation": pc.equation("rho(r, dv)")},
        "corr_z_dv_all": float(np.corrcoef(z, dv[ok])[0, 1]), "corr_absz_dv": float(np.corrcoef(np.abs(z), dv[ok])[0, 1]),
    }


def fp_check(fld: Field, st: pl.DataFrame, tr: np.ndarray, vf_info: dict, seed: int = 0) -> list[dict]:
    """2-D (log price, log vol) Fokker-Planck over 24h from three vol levels vs (a) Euler MC of the same SDE, (b) the
    full-field Monte-Carlo from in-sample 00:00 origins near that vol level, (c) the empirical 24h returns there."""
    b = vf_info["binned"]
    vc, n = np.asarray(b["v_center"]), np.asarray(b["n"], float)
    m1, Dv = np.asarray(b["drift_per_h"]), np.asarray(b["D_v_per_h"])
    good = np.isfinite(m1) & np.isfinite(Dv)
    cm = np.polyfit(vc[good], m1[good], 3, w=np.sqrt(n[good]))
    cdv = np.polyfit(vc[good], np.log(Dv[good]), 2, w=np.sqrt(n[good]))
    v = 0.5 * np.log(st["vf"].to_numpy())
    r = st["ret"].to_numpy().astype(float)
    c2 = float(np.mean((st["r_next"].to_numpy()[tr]) ** 2 / st["vf"].to_numpy()[tr]))
    sigma = lambda vv: np.exp(vv) * math.sqrt(c2)
    mu_v = lambda vv: np.polyval(cm, np.clip(vv, vc[good].min(), vc[good].max()))
    D_v = lambda vv: np.exp(np.polyval(cdv, np.clip(vv, vc[good].min(), vc[good].max())))
    R24 = fs.fwd_sum(r, 24)
    org = np.flatnonzero(tr & fs.origin_mask(st, 24) & np.isfinite(R24))
    out = []
    for q in (0.1, 0.5, 0.9):
        v0 = float(np.quantile(v[tr], q))
        span = 1.2 * math.sqrt(c2) * math.exp(v0 + 0.6) * math.sqrt(24) * 6
        xg = np.linspace(-span, span, 481)
        vg = np.linspace(vc[good].min() - 0.3, vc[good].max() + 0.3, 61)
        p0 = np.zeros((len(xg), len(vg)))
        p0[240, int(np.argmin(np.abs(vg - v0)))] = 1.0
        P = fp2d_logprice_vol(sigma, mu_v, D_v, xg, vg, p0, 24.0)
        q_fp = cdf_quantiles(xg, P.sum(1), QS)
        q_mc = np.quantile(mc_logprice_vol(sigma, mu_v, D_v, v0, 24.0, 20000, 12, seed), QS)
        near = org[np.abs(v[org] - v0) < 0.08]
        near = near[np.linspace(0, len(near) - 1, min(len(near), 150)).astype(int)] if len(near) else near
        q_emp = np.quantile(R24[near], QS) if len(near) > 20 else np.full(5, np.nan)
        if len(near):
            sim = simulate(fld, origins_from_state(st, near, fld.profile, 24), 24, 300, seed=seed + 1)
            q_fld = np.quantile(sim["terminal"].ravel(), QS)
        else:
            q_fld = np.full(5, np.nan)
        out.append({"v_quantile": q, "v0": v0, "sd_1h_pct": 100 * float(sigma(v0)), "n_origins": len(near),
                    "fp_pct": (100 * q_fp).round(3).tolist(), "sde_mc_pct": (100 * q_mc).round(3).tolist(),
                    "field_mc_pct": (100 * q_fld).round(3).tolist(), "empirical_pct": (100 * q_emp).round(3).tolist()})
    return out


def delta_check(fld: Field, as_of: datetime) -> dict:
    """Exploratory (no verdict): does signed flow (Bybit flow_bars, from 2026-05-11) relate to the field's residuals?"""
    d = flow_delta(as_of)
    if d.height == 0:
        return {"available": False}
    st = build_state(load_frame(as_of), fld.profile, fld.cfg)
    j = st.join(d, on="event_ts", how="inner").filter(pl.col("delta24").is_not_null() & pl.col("r_next").is_not_null())
    X = coords_matrix(j)
    mu, D, p = fld.params(X, np.log(j["vf"].to_numpy()))
    z = (j["r_next"].to_numpy() - mu) / np.sqrt(D)
    dl = j["delta24"].to_numpy()
    d1 = j["delta"].to_numpy()
    dls = (dl - dl.mean()) / dl.std()
    out = {"available": True, "from": str(j["event_ts"].min()), "to": str(j["event_ts"].max()), "n": j.height,
           "note": "Bybit flow_bars 1h; the lake's Binance bars carry no taker-buy volume (bars schema: volume/turnover), so delta history is 6 months only"}
    for name, y in (("z_next (direction)", z), ("z_next^2 - 1 (vol)", z * z - 1), ("|z_next| > 3 (tail)", (np.abs(z) > 3).astype(float))):
        for xn, x in (("delta24", dls), ("|delta24|", (np.abs(dl) - np.abs(dl).mean()) / np.abs(dl).std()), ("delta_1h", (d1 - d1.mean()) / d1.std())):
            b = np.polyfit(x, y, 1)[0]
            m, t, _ = newey_west_t((x - x.mean()) * (y - y.mean()) / max(np.var(x), 1e-12))
            out[f"{y_short(name)}~{xn}"] = {"slope": float(b), "t_nw": float(t)}
    return out


def y_short(name: str) -> str:
    return name.split(" ")[0]


# ------------------------------------------------------------------ verdicts


def fld1_verdict(rows: dict, holm_p: dict, seg: str) -> str:
    good = sum(1 for h in fs.HORIZONS if rows[h]["diff"] > 0 and rows[h]["t"] > 2 and holm_p[f"{seg}:FLD1:h{h}"] < 0.05)
    bad = sum(1 for h in fs.HORIZONS if not (rows[h]["diff"] > 0))
    return "supports" if good >= 2 else ("contradicts" if bad >= 2 else "inconclusive")


def fld2_verdict(br: dict, dr: dict) -> str:
    sup = all(br[h]["ci90"][1] <= 0.005 and abs(dr[h]["t"]) < 2 for h in (1, 24))
    con = any(br[h]["ci90"][0] > 0 or dr[h]["t"] > 2 for h in (1, 24))
    return "supports" if sup else ("contradicts" if con else "inconclusive")


def fld3_verdict(r1: dict, hp: float) -> str:
    if r1["diff"] > 0 and r1["t"] > 2 and hp < 0.05:
        return "supports"
    return "contradicts" if not (r1["diff"] > 0) else "inconclusive"


# ------------------------------------------------------------------ main analysis


def analyse(as_of: datetime, record: bool = False, log=print) -> dict:
    from ai_trader.research.hypotheses import record as rec_fn
    from ai_trader.research.hypotheses import results

    cut = knowledge_cutoff()
    ev = _load_months()
    fits = _load_fits()
    models = fs._models_dir()
    cut_path = models / f"btc_field_cutoff_{cut:%Y-%m-%d}.json"
    fld_cut = Field.load(cut_path)
    # FLD1 / FLD2 tables
    out: dict = {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "as_of": as_of.isoformat(), "cutoff": cut.isoformat(),
                 "segments": {}, "n_months": len(fits)}
    for seg in ("llm_in_sample", "clean_holdout"):
        e = _seg(ev, seg)
        s = {"fld1": {}, "vs_B1": {}, "field_gauss_vs_B0": {}, "drift": {}, "brier": {}, "pit": {}}
        for h in fs.HORIZONS:
            s["fld1"][h] = fs.paired(e, "field", "B0", h)
            s["vs_B1"][h] = fs.paired(e, "field", "B1", h)
            s["drift"][h] = fs.paired(e, "field", "field_nodrift", h)
            s["brier"][h] = fs.brier_skill(e, "field", h)
            s["pit"][h] = {m: fs.pit_table(e, m, h) for m in ("field", "B0", "B1")}
        s["field_gauss_vs_B0"][1] = fs.paired(e, "field_gauss", "B0", 1)
        s["B1_vs_B0"] = {h: fs.paired(e, "B1", "B0", h) for h in fs.HORIZONS}
        out["segments"][seg] = s
    # FLD3
    f3 = fs.run_fld3(as_of, log=log)
    out["fld3"] = {h: fs.paired(f3, "online", "frozen", h) for h in fs.HORIZONS}
    out["fld3_pit"] = {h: {m: fs.pit_table(f3, m, h) for m in ("frozen", "online")} for h in fs.HORIZONS}
    try:
        out["fld3_coef_drift"] = json.loads((fs.SCR / "fld3_coef_drift.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    # Holm
    pv = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        for h in fs.HORIZONS:
            pv[f"{seg}:FLD1:h{h}"] = out["segments"][seg]["fld1"][h]["p"]
    ins = fs.holm({k: v for k, v in pv.items() if k.startswith("llm")})
    hold_p = {k: v for k, v in pv.items() if k.startswith("clean")} | {f"clean_holdout:FLD3:h{h}": out["fld3"][h]["p"] for h in fs.HORIZONS}
    hp = ins | fs.holm(hold_p)
    out["holm"] = hp
    out["verdicts"] = {
        "H-FLD1": {seg: fld1_verdict(out["segments"][seg]["fld1"], hp, seg) for seg in ("llm_in_sample", "clean_holdout")},
        "H-FLD2": {seg: fld2_verdict(out["segments"][seg]["brier"], out["segments"][seg]["drift"]) for seg in ("llm_in_sample", "clean_holdout")},
        "H-FLD3": {"clean_holdout": fld3_verdict(out["fld3"][1], hp["clean_holdout:FLD3:h1"])},
    }
    # equations, fit history
    out["field_cutoff"] = _field_summary(fld_cut)
    out["fit_history"] = [{"month": f["month"], "selected_at": f.get("selected_at"), "n_train": f["field"]["meta"]["n_train"],
                           "jump_share": f["field"]["meta"]["jump_share"],
                           "n_terms": {k: len(f["field"]["components"][k]["terms"]) for k in ("mu", "logD", "loglam")},
                           "kou": f["field"]["kou"], "base_params": {k: f["base_params"][k] for k in ("B0_1", "B0_24")}} for f in fits]
    out["selection_history"] = {f["month"]: f["diag"].get("support") for f in fits if f.get("selected_at") == f["month"]}
    out["selection_probs_cutoff"] = next((f["diag"] for f in fits if f["month"] == f"{cut:%Y-%m}"), {})
    # descriptive vol field + FP check on the cutoff training sample
    frame = load_frame(as_of)
    st = build_state(frame, fld_cut.profile, fld_cut.cfg)
    kts = fs._ts_np(st["known_ts"])
    k_next = np.r_[kts[1:], np.datetime64("2100-01-01", "ms")]
    tr = (kts >= fs._np(fs.TRAIN_START)) & (k_next <= fs._np(cut)) & st["r_next"].is_not_null().to_numpy()
    log("vol field ...")
    vf_info = vol_field(fld_cut, st, tr)
    out["vol_field"] = vf_info
    fld_cut.vol_field = {k: vf_info[k] for k in ("coordinate", "ou_v", "ou_x1")} | {k: {"equation": vf_info[k]["equation"], "terms": vf_info[k]["terms"]}
                                                                                    for k in ("mu_v", "D_v", "rho_rv")}
    fld_cut.save(cut_path)
    log("FP check ...")
    out["fp_check"] = fp_check(fld_cut, st, tr, vf_info)
    log("delta check ...")
    out["delta_check"] = delta_check(fld_cut, as_of)
    # latest field (support of the cutoff selection, coefficients through as_of) and cards
    log("latest fit + cards ...")
    prev = next(f for f in fits if f["month"] == f"{cut:%Y-%m}")
    fld_now, st_now, tr_now, _ = fs.fit_field_at(frame, as_of, prev=prev, select=False)
    fld_now.vol_field = fld_cut.vol_field
    fld_now.meta["support_from"] = f"stability selection at {cut:%Y-%m-%d}"
    now_path = fld_now.save(models / f"btc_field_{as_of:%Y-%m-%d}.json")
    out["field_latest"] = _field_summary(fld_now)
    out["cards"] = _cards(fld_now, st_now)
    _write_index(models, cut_path, now_path, fits)
    # record
    out["recorded"] = []
    if record:
        prior = results()
        done = {(r["id"], r["segment"]) for r in prior}
        data = f"lake BTCUSDT 1h bybit, monthly refits {fs.EVAL_START:%Y-%m}..{as_of:%Y-%m}; research/reports/btc_field_{fs.REPORT_DATE}.md"
        for hid, segs in out["verdicts"].items():
            for seg, v in segs.items():
                if (hid, seg) in done:
                    continue
                rec_fn(hid, seg, v, _metrics(out, hid, seg), note=_note(hid, seg, out), data=data)
                out["recorded"].append([hid, seg, v])
    rep = load().root / "research" / "reports"
    (rep / f"btc_field_{fs.REPORT_DATE}.json").write_text(json.dumps(_jsonable(out), indent=1, default=float), encoding="utf-8")
    (rep / f"btc_field_{fs.REPORT_DATE}.md").write_text(render_md(out), encoding="utf-8")
    return out


def _metrics(out, hid, seg):
    if hid == "H-FLD1":
        return {f"h{h}": out["segments"][seg]["fld1"][h] | {"p_holm": out["holm"][f"{seg}:FLD1:h{h}"]} for h in fs.HORIZONS}
    if hid == "H-FLD2":
        s = out["segments"][seg]
        return {f"h{h}": {"brier_skill": s["brier"][h].get("bss"), "ci90": s["brier"][h].get("ci90"), "drift_diff": s["drift"][h]["diff"],
                          "drift_t": s["drift"][h]["t"]} for h in (1, 4, 24)}
    return {f"h{h}": out["fld3"][h] | {"p_holm": out["holm"][f"clean_holdout:FLD3:h{h}"]} for h in fs.HORIZONS}


def _note(hid, seg, out):
    if hid == "H-FLD1":
        b1 = out["segments"][seg]["vs_B1"]
        return "field vs log-HAR-t + constant Kou; vs the calendar baseline B1 (report only): " + ", ".join(
            f"h{h} {b1[h]['diff']:+.4f} (t {b1[h]['t']:.2f})" for h in fs.HORIZONS)
    if hid == "H-FLD2":
        return "supports needs Brier CI upper <= 0.005 and |t| < 2 for the drift term at 1h and 1d"
    return "frozen = cutoff fit; online = RLS with forgetting (memory 1440 h); primary 1h"


def _field_summary(fld: Field) -> dict:
    def terms(p):
        return [{"term": mono_name(m, p.names), "coef": float(c), "se": float(s), "t": float(c / s) if s > 0 else None}
                for m, c, s in zip(p.monos, p.coef, p.se, strict=True)]

    return {"meta": fld.meta, "equations": {"logD": fld.logD.equation("D", 3), "mu": fld.mu.equation("mu/sqrt(D)", 4),
                                            "loglam": fld.loglam.equation("p_jump", 3)},
            "terms": {"logD": terms(fld.logD), "mu": terms(fld.mu), "loglam": terms(fld.loglam)}, "kou": fld.kou.to_dict(), "t_dof": fld.t_dof,
            "standardizer": fld.std.to_dict()}


def _cards(fld: Field, st: pl.DataFrame) -> list[dict]:
    st = st.filter(pl.col("known_ts") <= st["known_ts"].max())
    n = st.height
    picks = {"now": n - 1}
    hr = st["close_ts"].dt.hour().to_numpy()
    wd = st["close_ts"].dt.weekday().to_numpy()
    fl = st["flag"].to_numpy()
    for name, mask in (("quiet Saturday 06 UTC", (wd == 6) & (hr == 6)), ("weekday US open 13 UTC", (wd <= 5) & (hr == 13)), ("1h after a jump bar", fl)):
        k = np.flatnonzero(mask[: n - 1])
        if len(k):
            picks[name] = int(k[-1])
    out = []
    for name, i in picks.items():
        c = card(fld, st, i, n_paths=4000, seed=i % 997)
        c["label"] = name
        out.append(c)
    return out


def _write_index(models: Path, cut_path: Path, now_path: Path, fits: list[dict]) -> None:
    hist = [{"month": f["month"], "logD": f["field"]["components"]["logD"]["terms"], "mu": f["field"]["components"]["mu"]["terms"],
             "loglam": f["field"]["components"]["loglam"]["terms"], "names": f["field"]["components"]["logD"]["names"], "kou": f["field"]["kou"],
             "standardizer": f["field"]["state"]["standardizer"]} for f in fits]
    for h in hist:  # keep it small: no covariance matrices in the history
        for k in ("logD", "mu", "loglam"):
            h[k] = [{"exponents": t["exponents"], "coef": t["coef"], "se": t["se"]} for t in h[k]]
    (models / "btc_field_monthly.json").write_text(json.dumps({"schema": "field.monthly.v1", "fits": hist}), encoding="utf-8")
    idx = {"schema": "field.index.v1", "frozen_cutoff": cut_path.name, "latest": now_path.name, "monthly": "btc_field_monthly.json",
           "loader": "ai_trader.field.model.Field.load(path)", "report": f"research/reports/btc_field_{fs.REPORT_DATE}.md"}
    (models / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float | np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


# ------------------------------------------------------------------ markdown


def _f(x, d=4, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render_md(o: dict) -> str:
    from ai_trader.research.field_text import narrative, rule_candidates, schema_doc

    L = [f"# BTC probability field (H-FLD1-3), {fs.REPORT_DATE}", "",
         f"_Generated {o['generated']} by `ai_trader.research.field_report` (agent F1; program research/plans/2026-11-13_btc_field_program.md). "
         f"Data: lake BTCUSDT 1h (Bybit), {o['n_months']} monthly refits from {fs.EVAL_START:%Y-%m}; knowledge cutoff {o['cutoff'][:10]}._", ""]
    L += narrative(o)
    L += ["", "## Verdicts", "", "| Hypothesis | llm_in_sample | clean_holdout |", "|---|---|---|"]
    for hid, v in o["verdicts"].items():
        L.append(f"| {hid} | {v.get('llm_in_sample', '-')} | {v.get('clean_holdout', '-')} |")
    L += ["", "## H-FLD1: log score, field minus log-HAR-t + constant Kou (B0)", "",
          "Per-origin mean difference in log predictive density (nats); t = Newey-West on the daily means; Holm within FLD per segment.", "",
          "| Segment | h | n origins | days | field | B0 | diff | t | Holm p | vs B1 (calendar) diff (t) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seg, s in o["segments"].items():
        for h in fs.HORIZONS:
            r, b1 = s["fld1"][h], s["vs_B1"][h]
            L.append(f"| {seg} | {h}h | {r['n']} | {r.get('days', '-')} | {_f(r.get('mean_a'))} | {_f(r.get('mean_b'))} | {_f(r['diff'])} | "
                     f"{_f(r['t'], 2)} | {_f(o['holm'][f'{seg}:FLD1:h{h}'], 3, False)} | {_f(b1['diff'])} ({_f(b1['t'], 2)}) |")
    L += ["", "Secondary (not a verdict): 1h field with a Gaussian non-jump branch (same fit) vs B0: " + "; ".join(
        f"{seg} {_f(s['field_gauss_vs_B0'][1]['diff'])} (t {_f(s['field_gauss_vs_B0'][1]['t'], 2)})" for seg, s in o["segments"].items()) +
          ". Calendar baseline B1 vs B0: " + "; ".join(f"{seg} " + ", ".join(f"h{h} {_f(s['B1_vs_B0'][h]['diff'])} (t {_f(s['B1_vs_B0'][h]['t'], 2)})"
                                                                              for h in fs.HORIZONS) for seg, s in o["segments"].items()) + ".", ""]
    L += ["## Calibration: PIT histograms (10 bins) and central-interval coverage", "",
          "| Segment | h | model | coverage 50 / 80 / 90 / 95 | P(PIT<.05) | P(PIT>.95) | PIT histogram |", "|---|---|---|---|---|---|---|"]
    for seg, s in o["segments"].items():
        for h in fs.HORIZONS:
            for m, t in s["pit"][h].items():
                if not t:
                    continue
                c = t["coverage"]
                L.append(f"| {seg} | {h}h | {m} | {c['50']:.3f} / {c['80']:.3f} / {c['90']:.3f} / {c['95']:.3f} | {t['tail_lo_5']:.3f} | {t['tail_hi_5']:.3f} | "
                         f"{' '.join(f'{x:.3f}' for x in t['hist'])} |")
    L += ["", "## H-FLD2: drift (null)", "", "| Segment | h | Brier skill of P(R>0) vs base rate | 90% CI (10-day blocks) | drift-on minus drift-off log score | t |",
          "|---|---|---|---|---|---|"]
    for seg, s in o["segments"].items():
        for h in fs.HORIZONS:
            b, d = s["brier"][h], s["drift"][h]
            ci = b.get("ci90", [float("nan")] * 2)
            L.append(f"| {seg} | {h}h | {_f(b.get('bss'), 5)} | [{_f(ci[0], 5)}, {_f(ci[1], 5)}] | {_f(d['diff'], 5)} | {_f(d['t'], 2)} |")
    L += ["", "## H-FLD3: online (RLS, forgetting 1 - 1/1440) vs frozen at the cutoff, clean_holdout", "",
          "| h | n | frozen | online | diff | t | Holm p |", "|---|---|---|---|---|---|---|"]
    for h in fs.HORIZONS:
        r = o["fld3"][h]
        L.append(f"| {h}h | {r['n']} | {_f(r.get('mean_b'))} | {_f(r.get('mean_a'))} | {_f(r['diff'])} | {_f(r['t'], 2)} | {_f(o['holm'][f'clean_holdout:FLD3:h{h}'], 3, False)} |")
    fc = o["field_cutoff"]
    L += ["", "## The discovered field (fit at the cutoff, training 2020-09-01 .. 2026-06-30)", "",
          "Coordinates are standardised (x~ = (x - mean) / sd; means and sds below). Law per hour: r = mu + sqrt(D) (e + 1[jump] K).", "",
          "```", fc["equations"]["logD"], fc["equations"]["mu"], fc["equations"]["loglam"], "```", "",
          f"Kou jump law (diffusion-sd units): p_up {fc['kou']['p_up']:.3f}, mean up {fc['kou']['mean_up_sd']:.2f} sd, mean down {fc['kou']['mean_down_sd']:.2f} sd; "
          f"jump share of training bars {fc['meta']['jump_share']:.3f}; non-jump branch Student-t df {fc['t_dof']:.1f} (unit variance).", "",
          "| component | term | coef | se | t |", "|---|---|---|---|---|"]
    for k in ("logD", "loglam", "mu"):
        for t in fc["terms"][k]:
            L.append(f"| {k} | {t['term']} | {t['coef']:+.4f} | {t['se']:.4f} | {_f(t['t'], 1)} |")
    sd = fc["standardizer"]
    L += ["", "Standardisation: " + ", ".join(f"{n} mean {m:.4g} sd {s:.4g}" for n, m, s in zip(sd["names"], sd["mean"], sd["sd"], strict=True)), ""]
    sel = o.get("selection_probs_cutoff", {})
    if sel:
        L += ["Stability-selection probabilities at the cutoff (>= 0.3 shown; threshold 0.7):", ""]
        for k in ("sel_prob_D", "sel_prob_lam", "sel_prob_mu"):
            if k in sel:
                L.append(f"- {k[9:]}: " + ", ".join(f"{t} {p:.2f}" for t, p in sorted(sel[k].items(), key=lambda kv: -kv[1])[:20]))
        L.append("")
    L += ["Support history (stability selection points): " + "; ".join(f"{m}: D {len(v['D'])}, lam {len(v['lam'])}, mu {len(v['mu'])} terms"
                                                                         for m, v in o["selection_history"].items() if v), ""]
    vf = o["vol_field"]
    L += ["## Vol field (descriptive): Kramers-Moyal of v = 0.5 ln vf", "",
          f"OU fit of v with exact finite-dt correction: theta {vf['ou_v']['theta_per_h']:.4f}/h (naive {vf['ou_v']['theta_naive']:.4f}), half-life "
          f"{vf['ou_v']['half_life_h']:.1f} h, D {vf['ou_v']['D']:.3g}/h. x1 (vol shock): half-life {vf['ou_x1']['half_life_h']:.1f} h. "
          f"corr(z_next, dv) {vf['corr_z_dv_all']:+.3f}; corr(|z_next|, dv) {vf['corr_absz_dv']:+.3f} (vol responds to the size, not the sign).", "",
          "```", vf["mu_v"]["equation"], vf["D_v"]["equation"], vf["rho_rv"]["equation"], "```", "",
          "| v bin center | n | drift /h | D_v /h | corr(dv, z) |", "|---|---|---|---|---|"]
    b = vf["binned"]
    for i in range(len(b["v_center"])):
        L.append(f"| {b['v_center'][i]:.3f} | {int(b['n'][i])} | {_f(b['drift_per_h'][i], 5)} | {_f(b['D_v_per_h'][i], 6, False)} | {_f(b['corr_r_dv'][i], 3)} |")
    L += ["", "## Fokker-Planck check (2-D log price x log vol, 24h)", "",
          "Quantiles q05 / q25 / q50 / q75 / q95 of the 24h log return (%). FP: finite-volume solution of the reduced 2-D field (binned KM drift / diffusion of v, sigma = c e^v); "
          "SDE MC: Euler MC of the same SDE (checks the solver); field MC: the full field's Monte-Carlo from in-sample 00:00 origins near that vol level; empirical: realised.", "",
          "| vol level (quantile of v) | 1h sd % | n origins | FP | SDE MC | field MC | empirical |", "|---|---|---|---|---|---|---|"]
    for r in o["fp_check"]:
        fmt = lambda a: " / ".join("-" if (x is None or not np.isfinite(x)) else f"{x:+.2f}" for x in a)
        L.append(f"| q{int(100 * r['v_quantile'])} | {r['sd_1h_pct']:.2f} | {r['n_origins']} | {fmt(r['fp_pct'])} | {fmt(r['sde_mc_pct'])} | {fmt(r['field_mc_pct'])} | {fmt(r['empirical_pct'])} |")
    dc = o["delta_check"]
    L += ["", "## Signed flow (delta), exploratory", ""]
    if dc.get("available"):
        L.append(f"{dc['note']}. Window {dc['from'][:10]} .. {dc['to'][:10]}, n = {dc['n']} hours; slopes of standardised field residuals on standardised flow (NW t):")
        L.append("")
        L += [f"- {k}: slope {v['slope']:+.4f}, t {v['t_nw']:+.2f}" for k, v in dc.items() if isinstance(v, dict)]
    else:
        L.append("No flow data.")
    L += ["", "## Distribution cards (field fitted through " + o["as_of"][:16] + ", support from the cutoff selection)", ""]
    for c in o["cards"]:
        L += [f"### {c['label']}", "", "```", c["text"], "```", ""]
    L += schema_doc()
    L += ["", "## Rule candidates", "", "```yaml", rule_candidates(o), "```", ""]
    return "\n".join(L)


_ = (glm_fit, Path)
