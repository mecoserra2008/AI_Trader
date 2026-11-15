/* Percentile of a move inside a 7-quantile predictive distribution (taus .05 .10 .25 .50 .75 .90 .95).
   The same algorithm as src/ai_trader/livedash/pct.py (tested equal): z(x) piecewise linear in probit space through
   (q_k, Phi^-1(tau_k)), the outermost segment extended in the tails; F(x) = Phi(z(x)). Works in the browser
   (window.PCT) and in node (module.exports). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PCT = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
  var QN = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"];
  var ZK = [-1.6448536269514726, -1.2815515655446008, -0.6744897501960817, 0, 0.6744897501960817, 1.2815515655446008, 1.6448536269514715];
  var EPS = 1e-9;

  // Normal CDF via erfc (W. J. Cody's rational approximations as in Numerical Recipes erfcc; |err| < 1.2e-7)
  function erfc(x) {
    var z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  function cdf(z) { return 0.5 * erfc(-z / Math.SQRT2); }

  function knots(q) {
    var v = Array.isArray(q) ? q.slice() : QN.map(function (k) { return q[k]; });
    if (v.length !== 7 || v.some(function (x) { return x === null || x === undefined || !isFinite(x); })) return null;
    for (var i = 1; i < 7; i++) v[i] = Math.max(+v[i], v[i - 1] + EPS);
    return v.map(Number);
  }
  function zscore(x, q) {
    var k = knots(q);
    if (!k) return NaN;
    var i;
    if (x <= k[0]) i = 0;
    else if (x >= k[6]) i = 5;
    else { i = 0; for (var j = 0; j < 6; j++) if (k[j] <= x) i = j; }
    return ZK[i] + (x - k[i]) * (ZK[i + 1] - ZK[i]) / (k[i + 1] - k[i]);
  }
  function percentile(x, q) { return cdf(zscore(x, q)); }
  function scaled(q, frac) {
    var k = knots(q), s = Math.sqrt(Math.max(0, Math.min(1, frac)));
    return k ? k.map(function (v) { return v * s; }) : null;
  }
  function describe(p) {
    if (p < 0.05) return "an extreme move down, below the network's 90% band";
    if (p < 0.15) return "an unusually strong move down";
    if (p < 0.30) return "a softer hour than the network's median, still ordinary";
    if (p <= 0.70) return "an ordinary hour, close to the middle of what it expected";
    if (p <= 0.85) return "a firmer hour than the network's median, still ordinary";
    if (p <= 0.95) return "an unusually strong move up";
    return "an extreme move up, above the network's 90% band";
  }
  function ordinal(n) {
    var s = (n % 100 >= 10 && n % 100 <= 20) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th");
    return n + s;
  }
  function sentence(p, horizon) {
    var k = Math.min(99, Math.max(1, Math.round(100 * p)));
    return "BTC is at the " + ordinal(k) + " percentile of what the network expected for this " + (horizon || "hour") + ": " + describe(p) + ".";
  }
  return { TAUS: TAUS, QN: QN, ZK: ZK, cdf: cdf, zscore: zscore, percentile: percentile, scaled: scaled,
           describe: describe, ordinal: ordinal, sentence: sentence };
});
