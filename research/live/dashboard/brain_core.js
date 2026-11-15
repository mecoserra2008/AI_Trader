/* Brain core: the frozen 1h PNN forward pass (float64) and the intra-hour nowcast recipe (brain_recipe.v1).
   Pure functions, no DOM: loaded by brain.html and by tests/test_brain.py through node (module.exports).
   Mirrors src/ai_trader/livedash/brain.py (forward, partial_state, nowcast_inputs, remaining_pct); tests check equality. */
(function (root, factory) {
  var api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BrainCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var QNAMES = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"];
  var HOUR_MS = 3600000, F_WARM = 5 / 60, U_FLOOR = 1e-4, JUMP_Z = 4.0;

  function silu(x) { return x / (1 + Math.exp(-x)); }
  function softplus(x) { return x > 20 ? x : Math.log1p(Math.exp(x)); }

  function standardise(net, raw) {
    var n = raw.length, x = new Float64Array(n), c = net.clip;
    for (var i = 0; i < n; i++) {
      var v = raw[i];
      var z = (v === null || v === undefined || !isFinite(v)) ? 0 : (v - net.std_mu[i]) / net.std_sd[i];
      if (!isFinite(z)) z = 0;
      x[i] = z > c ? c : (z < -c ? -c : z);
    }
    return x;
  }

  /* One KAN layer: E[o][i] = w_res[o][i] silu(x_i) + sum_d coef[o][i][d] T_d(tanh(scale x_i)); node = bias + sum_i E. */
  function layer(L, x, scale, D) {
    var nin = x.length, nout = L.bias.length, B = new Array(nin), S = new Float64Array(nin);
    for (var i = 0; i < nin; i++) {
      var u = Math.tanh(scale * x[i]), t = [u, 2 * u * u - 1];
      for (var d = 2; d < D; d++) t.push(2 * u * t[d - 1] - t[d - 2]);
      B[i] = t; S[i] = silu(x[i]);
    }
    var E = new Array(nout), node = new Float64Array(nout);
    for (var o = 0; o < nout; o++) {
      var row = new Float64Array(nin), acc = L.bias[o], co = L.coef[o], wo = L.w_res[o];
      for (var j = 0; j < nin; j++) {
        var e = wo[j] * S[j], cj = co[j], bj = B[j];
        for (var k = 0; k < D; k++) e += cj[k] * bj[k];
        row[j] = e; acc += e;
      }
      E[o] = row; node[o] = acc;
    }
    return { E: E, node: node };
  }

  function quantilesFromRaw(o) {
    var q50 = o[3], q75 = q50 + softplus(o[4]), q90 = q75 + softplus(o[5]), q95 = q90 + softplus(o[6]);
    var q25 = q50 - softplus(o[2]), q10 = q25 - softplus(o[1]), q05 = q10 - softplus(o[0]);
    return [q05, q10, q25, q50, q75, q90, q95];
  }

  /* Full decomposition: x (standardised), per seed {enc (E 4x19), z, head (E 7x4), o, q}, ensemble q (model units). */
  function forward(net, raw) {
    var x = standardise(net, raw), seeds = [], q = [0, 0, 0, 0, 0, 0, 0];
    for (var s = 0; s < net.seeds.length; s++) {
      var a = layer(net.seeds[s].enc, x, net.scale, net.degree);
      var b = layer(net.seeds[s].head, a.node, net.scale, net.degree);
      var qs = quantilesFromRaw(b.node);
      for (var k = 0; k < 7; k++) q[k] += qs[k] / net.seeds.length;
      seeds.push({ enc: a.E, z: a.node, head: b.E, o: b.node, q: qs });
    }
    return { x: x, seeds: seeds, q: q };
  }

  /* Partial-hour state from data with S <= t <= now only. ticks [[ts_ms, price]], candles [[start_ms, o, h, l, c, ...]]
     (a candle counts only once closed: start + 60 s <= now, at time start + 60 s). */
  function partialState(p0, S_ms, now_ms, ticks, candles) {
    var ev = [], i;
    candles = candles || []; ticks = ticks || [];
    for (i = 0; i < candles.length; i++) {
      var c = candles[i], st = +c[0];
      if (st >= S_ms && st + 60000 <= now_ms && st < S_ms + HOUR_MS) ev.push([st + 60000, +c[4], +c[2], +c[3]]);
    }
    for (i = 0; i < ticks.length; i++) {
      var ts = +ticks[i][0], p = +ticks[i][1];
      if (ts >= S_ms && ts <= now_ms && ts <= S_ms + HOUR_MS) ev.push([ts, p, p, p]);
    }
    ev.sort(function (a, b) { return a[0] - b[0]; });
    var P = +p0, H = +p0, L = +p0, t = S_ms, n = 0;
    for (i = 0; i < ev.length; i++) {
      var e = ev[i];
      if (!(e[1] > 0 && e[2] > 0 && e[3] > 0)) continue;
      P = e[1]; t = e[0]; n++;
      H = Math.max(H, e[2], e[1]); L = Math.min(L, e[3], e[1]);
    }
    return { P: P, H: H, L: L, t_last_ms: t, n: n };
  }

  /* brain_recipe.v1 (see brain.py): returns [raw inputs, info]. */
  function nowcastInputs(base, state, now_ms) {
    var feats = base.features, x = [], idx = {}, i;
    for (i = 0; i < feats.length; i++) { idx[feats[i]] = i; x.push(base.inputs[i] === null ? NaN : +base.inputs[i]); }
    var f = (now_ms - base.S_ms) / HOUR_MS;
    f = Math.min(Math.max(f, 0), 1);
    var p0 = +base.p0;
    var info = { f: f, warm: f < F_WARM, P: state.P, H: state.H, L: state.L, r_p: Math.log(state.P / p0),
                 range: Math.log(state.H / state.L), u: null, J: 0 };
    if (f < F_WARM) return [x, info];
    var s2 = base.s2, u = info.r_p * info.r_p / (f * s2), uLast = Math.exp(x[idx.lu_1]);
    x[idx.lu_1] = Math.log(Math.max(u, U_FLOOR));
    x[idx.lu_m6] = Math.log(Math.max(Math.exp(x[idx.lu_m6]) + (u - uLast) / 6, 1e-12));
    x[idx.lu_m24] = Math.log(Math.max(Math.exp(x[idx.lu_m24]) + (u - uLast) / 24, 1e-12));
    var exc = Math.max(Math.log(state.H / p0), Math.log(p0 / state.L));
    var J = exc > JUMP_Z * Math.sqrt(s2) ? 1 : 0;
    if (J) { x[idx.jump24] = Math.max(x[idx.jump24], 1); x[idx.since_jump] = 1 / 168; }
    info.u = u; info.J = J; info.excursion = exc;
    return [x, info];
  }

  function remainingPct(q, base, f) {
    var k = 100 * Math.sqrt(base.s2) * Math.sqrt(Math.max(1 - f, 0));
    return q.map(function (v) { return k * v; });
  }

  /* The edge function itself, for tooltips and curves. */
  function edgeValue(L, o, i, x, scale, D) {
    var u = Math.tanh(scale * x), t = [u, 2 * u * u - 1];
    for (var d = 2; d < D; d++) t.push(2 * u * t[d - 1] - t[d - 2]);
    var e = L.w_res[o][i] * silu(x);
    for (var k = 0; k < D; k++) e += L.coef[o][i][k] * t[k];
    return e;
  }

  return { QNAMES: QNAMES, HOUR_MS: HOUR_MS, F_WARM: F_WARM, forward: forward, standardise: standardise, layer: layer,
           quantilesFromRaw: quantilesFromRaw, partialState: partialState, nowcastInputs: nowcastInputs,
           remainingPct: remainingPct, edgeValue: edgeValue, silu: silu, softplus: softplus };
});
