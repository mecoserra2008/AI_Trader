"""'What it's thinking': plain sentences from one line and its activations (deterministic text, no LLM).

  variance   the 1h sigma against its trailing-week mean, split into the scale input (sigma_ref = exp(lsig2 / 2))
             and the network's own multiplier; the top 3 inputs by |one-at-a-time effect| of moving each to its
             weekly mean (activations.week_ablation_dln_sigma), else by the training-mean ablation
  latent     how z moved since the previous line (latent sd units), where the dynamics put z next, last hour's error
  v1 vs v2   band widths, ACI alpha_t, the v2 jump-cluster floor
  coverage   the coverage alarms, worst gap first, plus any other alarm verbatim
  field      the field's touch probabilities for the hour
"""

from __future__ import annotations

import math
import re

LABELS = {
    "lsig2": "log realised-vol scale", "lfac": "vol factor", "fund_after": "post-funding hour", "fund_before": "pre-funding hour",
    "lu_1": "1h abs-return (log)", "lu_2": "2h-back abs-return", "lu_3": "3h-back abs-return", "lu_6": "6h abs-return",
    "lu_12": "12h abs-return", "lu_24": "24h abs-return", "lu_m6": "6h mean abs-return", "lu_m24": "24h mean abs-return",
    "jump24": "jump in last 24h", "since_jump": "time since jump", "har_d": "HAR daily vol", "har_w": "HAR weekly vol",
    "har_m": "HAR monthly vol", "dow_sin": "weekday (sin)", "dow_cos": "weekday (cos)",
}


def _pct(x: float) -> str:
    return f"{100 * (math.exp(x) - 1):+.0f}%"


def variance(act: dict, line: dict) -> str:
    a = act["horizons"]["1h"]
    at = a["attribution"]
    feats = a["features"]
    sig = line["net"]["1h"]["sigma_pct"]
    wk = at.get("week", {})
    if wk.get("n") and at.get("week_ablation_dln_sigma") is not None:
        eff = at["week_ablation_dln_sigma"]
        top = sorted(range(len(feats)), key=lambda i: -abs(eff[i]))[:3]
        parts = []
        for i in top:
            now, ref = a["raw"][i], wk["mean_raw"][i]
            parts.append(f"{feats[i]} ({LABELS.get(feats[i], feats[i])}) {now:+.2f} vs week {ref:+.2f}, {_pct(eff[i])} on sigma")
        return (f"Hourly sigma {sig:.2f}% vs its mean {wk['mean_sigma_pct']:.2f}% over the past week's {wk['n']} hourly lines ({_pct(wk['dln_sigma'])}). "
                f"The scale input lsig2 (sigma_ref = exp(lsig2/2)) accounts for {_pct(wk['dln_scale'])} and the network's own multiplier for {_pct(wk['dln_model'])}. "
                f"Top drivers of the multiplier (each input moved alone to its weekly mean): " + "; ".join(parts) + ".")
    eff = at["mean_ablation_sigma_pct"]
    top = sorted(range(len(feats)), key=lambda i: -abs(eff[i]))[:3]
    return f"Hourly sigma {sig:.2f}% (no weekly reference yet). Largest effects vs each input's training mean: " + "; ".join(
        f"{feats[i]} {eff[i]:+.3f} pp" for i in top) + "."


def latent(line: dict, prev: dict | None) -> str:
    lat = line.get("latent", {})
    z, zh, mu, sd = lat.get("z"), lat.get("zhat_next"), lat.get("zmu"), lat.get("zsd")
    if not z or not zh or not mu or not sd:
        return "No latent stored for this line."
    zs = [(v - m) / s for v, m, s in zip(z, mu, sd, strict=True)]
    nxt = [(v - m) / s for v, m, s in zip(zh, mu, sd, strict=True)]
    txt = "Latent (sd units from its training mean): [" + ", ".join(f"{v:+.2f}" for v in zs) + "]"
    pz = (prev or {}).get("latent", {}).get("z")
    if pz:
        d = [(a - b) / s for a, b, s in zip(z, pz, sd, strict=True)]
        j = max(range(len(d)), key=lambda i: abs(d[i]))
        txt += f"; since {prev['id']} the biggest move was z{j + 1} {d[j]:+.2f} sd (total {math.sqrt(sum(v * v for v in d)):.2f} sd)"
    dn = [b - a for a, b in zip(zs, nxt, strict=True)]
    j = max(range(len(dn)), key=lambda i: abs(dn[i]))
    rev = "back toward its mean" if zs[j] * dn[j] < 0 else "further from its mean"
    txt += ". The dynamics put the next step at [" + ", ".join(f"{v:+.2f}" for v in nxt) + f"], mainly z{j + 1} {dn[j]:+.2f} sd {rev}"
    le = lat.get("last_error")
    if le:
        better = "beat" if le["model_rmse"] < le["persistence_rmse"] else "lost to"
        txt += f". Last hour the latent model {better} persistence ({le['model_rmse']:.2f} vs {le['persistence_rmse']:.2f} sd)"
    g = lat.get("r2_gain_168")
    if g:
        txt += f"; 168h R^2 gain {g['gain']:+.3f}"
    return txt + "."


def versions(line: dict, v2: dict | None) -> str:
    if v2 is None:
        return "No v2 line for this hour (the v2 shadow journal is absent or not yet written)."
    a1, b2 = line["net"]["1h"]["aci"], v2["net"]["1h"]["band"]
    w = lambda b: b["hi_pct"] - b["lo_pct"]
    parts = [f"{L}% v1 [{a1[L]['lo_pct']:+.2f}, {a1[L]['hi_pct']:+.2f}] vs v2 [{b2[L]['lo_pct']:+.2f}, {b2[L]['hi_pct']:+.2f}] ({100 * (w(b2[L]) / w(a1[L]) - 1):+.0f}% width)"
             for L in ("80", "95") if L in a1 and L in b2 and w(a1[L]) > 0]
    al = f"alpha_t 80% v1 {a1['80']['alpha_t']:.3f} vs v2 {v2['net']['1h']['aci']['80']['alpha_t']:.3f}"
    fl = v2["net"]["1h"].get("floor", {})
    ftxt = (f"the jump-cluster floor is ACTIVE (sigma floor {fl.get('sigma_floor_pct', 0):.2f}%, applied to {', '.join(map(str, fl.get('applied', []))) or 'none'})"
            if fl.get("active") else "the jump-cluster floor is inactive")
    return "Same network, different conformal layer: " + "; ".join(parts) + f"; {al}; {ftxt}."


_COV = re.compile(r"coverage (\w+) (?:ACI|v2) (\d+)% over last (\d+): ([\d.]+) vs ([\d.]+)")


def coverage(alarms: list[dict]) -> str:
    cov, other = [], []
    for a in alarms:
        m = _COV.match(a["text"])
        if m:
            cov.append((float(m[5]) - float(m[4]), a["source"], m[1], m[2], m[3], float(m[4])))
        else:
            other.append(f"{a['source']}: {a['text']}")
    if not cov and not other:
        return "No alarms: every band's running coverage is inside its binomial 99% interval."
    s = ""
    if cov:
        cov.sort(reverse=True)
        s = f"{len(cov)} coverage alarm{'s' if len(cov) > 1 else ''}; the bands have been too narrow. Worst: " + "; ".join(
            f"{src} {h} {L}% band covered {100 * c:.1f}% of the last {n} (target {L}%)" for _, src, h, L, n, c in cov[:3]) + "."
    if other:
        s += (" " if s else "") + "Other: " + "; ".join(other[:4]) + "."
    return s


def field(line: dict) -> str | None:
    f = line.get("field", {}).get("horizons", {}).get("1h")
    if not f:
        return None
    t = f["p_touch"]
    return (f"Field ({line['field'].get('cell', '?')}): this hour touches +1% with {100 * t['+1%']:.0f}% and -1% with {100 * t['-1%']:.0f}%, "
            f"±2% with {100 * t['+2%']:.0f}/{100 * t['-2%']:.0f}%.")  # the field P(jump) is not shown (D3, H-WS05-6)


def thinking(act: dict, line: dict, prev: dict | None, v2: dict | None, alarms: list[dict]) -> list[dict]:
    out = [{"k": "variance", "text": variance(act, line)}, {"k": "latent", "text": latent(line, prev)},
           {"k": "versions", "text": versions(line, v2)}, {"k": "coverage", "text": coverage(alarms)}]
    f = field(line)
    if f:
        out.append({"k": "field", "text": f})
    return out


__all__ = ["coverage", "field", "latent", "thinking", "variance", "versions"]
