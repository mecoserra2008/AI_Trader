/* Cross-section v3 panel (xs.v3): relative strength across the 23 universe-L perps, CONTEXT ONLY, NOT A SIGNAL
   (R-XS-2 / R-XS-3: the ranking is mostly a low-vol ranking and its top-minus-bottom spread loses after costs).
   Reads /api/xs (the newest data/journal/live_xs_v3 line); optional live relative moves from Bybit's public websocket
   `tickers` topic. Read only: nothing here can trade. All times UTC. */
(function () {
  "use strict";

  var WS_URL = "wss://stream.bybit.com/v5/public/linear";
  var POLL_MS = 60000, PING_MS = 20000, DRAW_MIN_MS = 1000;
  var SECTORS = ["btc", "l1", "payments", "meme", "oracle", "exchange", "infra", "defi"];
  var X = { line: null, scores: [], h: "4h", px: {}, ws: null, wsOk: false, drawAt: 0, timer: null, err: "" };

  function $(id) { return document.getElementById(id); }
  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function fin(v) { return v !== null && v !== undefined && isFinite(v); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function sh(sym) { return sym.replace(/USDT$/, ""); }
  function bp(v, d) { return fin(v) ? ((v >= 0 ? "+" : "") + (+v).toFixed(d || 0)) : "-"; }
  function secColor(s) { return css("--sec-" + (SECTORS.indexOf(s) >= 0 ? s : "other")) || css("--ink-3"); }
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  var tip = null;
  function showTip(ev, html) {
    tip = tip || $("tip"); if (!tip) return;
    tip.innerHTML = html; tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight, x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
  }
  function hideTip() { if (tip) tip.hidden = true; }

  // ------------------------------------------------------------------ data
  function load() {
    return fetch("/api/xs", { cache: "no-store" }).then(function (r) {
      if (r.status === 404) throw new Error("the running server has no /api/xs yet (restart it)");
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; });
    }).then(function (j) {
      X.err = ""; X.scores = j.scores || [];
      var changed = !X.line || !j.line || X.line.id !== j.line.id;
      X.line = j.line; X.age = j.age_min;
      if (changed && X.line) subscribe();
      draw(true);
    }).catch(function (e) { X.err = e.message; draw(true); });
  }

  // ------------------------------------------------------------------ live relative moves (Bybit public tickers)
  function symbols() { return X.line ? X.line.tokens.filter(function (t) { return t.present; }).map(function (t) { return t.symbol; }) : []; }
  function subscribe() {
    var syms = symbols(); if (!syms.length) return;
    if (X.ws) { try { X.ws.onclose = null; X.ws.close(); } catch (e) { /* closed */ } }
    var ws;
    try { ws = new WebSocket(WS_URL); } catch (e) { X.wsOk = false; return; }
    X.ws = ws;
    ws.onopen = function () {
      X.wsOk = true;
      for (var i = 0; i < syms.length; i += 10) {
        ws.send(JSON.stringify({ op: "subscribe", args: syms.slice(i, i + 10).map(function (s) { return "tickers." + s; }) }));
      }
      ws._ping = setInterval(function () { if (ws.readyState === 1) ws.send(JSON.stringify({ op: "ping" })); }, PING_MS);
    };
    ws.onmessage = function (e) {
      var m; try { m = JSON.parse(e.data); } catch (x) { return; }
      if (m.topic && m.topic.indexOf("tickers.") === 0 && m.data && m.data.lastPrice) {
        X.px[m.topic.slice(8)] = { p: +m.data.lastPrice, t: Date.now() };
        draw(false);
      }
    };
    ws.onclose = function () { X.wsOk = false; clearInterval(ws._ping); setTimeout(function () { if (X.ws === ws) subscribe(); }, 5000); };
    ws.onerror = function () { try { ws.close(); } catch (x) { /* closed */ } };
  }
  // move since the forecast hour start (vs close_ref = close of bar S-1), minus the equal-weight mean of tokens with a price
  function liveRel() {
    var L = X.line; if (!L) return null;
    var lr = {}, n = 0, s = 0;
    L.tokens.forEach(function (t) {
      var q = X.px[t.symbol];
      if (t.present && q && fin(t.close_ref) && t.close_ref > 0 && Date.now() - q.t < 120000) { lr[t.symbol] = Math.log(q.p / t.close_ref); s += lr[t.symbol]; n++; }
    });
    if (n < 5) return null;
    var mu = s / n, out = {};
    for (var k in lr) out[k] = 1e4 * (lr[k] - mu);
    return { rel: out, n: n };
  }

  // ------------------------------------------------------------------ draw
  function draw(force) {
    if (!force && Date.now() - X.drawAt < DRAW_MIN_MS) { if (!X.timer) X.timer = setTimeout(function () { X.timer = null; draw(true); }, DRAW_MIN_MS); return; }
    X.drawAt = Date.now();
    var L = X.line;
    if (!L) { $("xs-meta").textContent = X.err ? "v3 unavailable: " + X.err : "no v3 line yet (the hourly run writes the first one at hh:03)"; return; }
    var S = Date.parse(L.forecast_hour), hh = X.h === "4h" ? 4 : 24, el = (Date.now() - S) / (hh * 3600000);
    $("xs-meta").textContent = L.id + " | " + L.n_present + "/" + L.n_symbols + " tokens | logged " + L.logged_at.slice(11, 16) + " UTC" +
      (fin(X.age) && X.age > 75 ? " | STALE (" + Math.round(X.age) + " min old)" : "") + (X.err ? " | " + X.err : "") +
      " | live feed " + (X.wsOk ? "on" : "off");
    var live = liveRel();
    drawTable(L, live, el);
    drawMap(L);
    drawAttention(L);
    drawFoot(L, live, el);
  }

  function drawTable(L, live, el) {
    var h = X.h, tok = L.tokens.filter(function (t) { return t.present; });
    tok.sort(function (a, b) { return a["rank_" + h] - b["rank_" + h]; });
    var ext = 0;
    tok.forEach(function (t) { var a = t.rel[h].aci_bp, q = t.rel[h].q_bp; ext = Math.max(ext, Math.abs(a[0]), Math.abs(a[1]), Math.abs(q[0]), Math.abs(q[2])); });
    ext = Math.ceil(ext / 50) * 50 || 100;
    var W = 260, H = 16, x0 = 4, x1 = W - 4, sx = function (v) { return x0 + (Math.max(-ext, Math.min(ext, v)) + ext) / (2 * ext) * (x1 - x0); };
    var fr = Math.sqrt(Math.max(0.02, Math.min(1, el)));
    var head = "<thead><tr><th>#</th><th>Token</th><th class='xs-band-h'>" + h + " rel. return vs EW basket, bp (80% ACI band, raw q10-q90, median" +
      (live ? ", live move" : "") + ")<span class='xs-axis'><span>" + (-ext) + "</span><span>0</span><span>+" + ext + "</span></span></th>" +
      "<th>Median</th><th>Band</th><th title='rank by low current volatility (-lsig2): what a free low-vol ranking would say'>Low-vol rank</th>" +
      "<th title='low-vol rank minus v3 rank: positive = v3 places the token higher than low vol alone'>v3 - low-vol</th>" +
      "<th title='1h |r| q90: v3 vs panel log-HAR-t (R-XS-1)'>1h |r| q90 v3 / HAR</th></tr></thead>";
    var rows = tok.map(function (t) {
      var r = t.rel[h], a = r.aci_bp, q = r.q_bp, c = secColor(t.sector), lv = live && live.rel[t.symbol];
      var svg = "<svg viewBox='0 0 " + W + " " + H + "' width='" + W + "' height='" + H + "' class='xs-bar' aria-hidden='true'>" +
        "<line x1='" + sx(0) + "' x2='" + sx(0) + "' y1='0' y2='" + H + "' class='xs-zero'/>" +
        "<rect x='" + sx(a[0]) + "' y='4' width='" + Math.max(1, sx(a[1]) - sx(a[0])) + "' height='8' rx='3' fill='" + c + "' fill-opacity='0.22'/>" +
        "<line x1='" + sx(q[0]) + "' x2='" + sx(q[2]) + "' y1='8' y2='8' stroke='" + c + "' stroke-width='2' stroke-linecap='round'/>" +
        "<line x1='" + sx(q[1]) + "' x2='" + sx(q[1]) + "' y1='2' y2='14' stroke='" + css("--ink") + "' stroke-width='2'/>";
      if (fin(lv)) {
        var inb = lv >= a[0] * fr && lv <= a[1] * fr;
        svg += "<rect x='" + sx(a[0] * fr) + "' y='6' width='" + Math.max(1, sx(a[1] * fr) - sx(a[0] * fr)) + "' height='4' rx='2' fill='none' stroke='" + css("--ink-3") + "' stroke-dasharray='2 2'/>" +
          "<circle cx='" + sx(lv) + "' cy='8' r='4' fill='" + (inb ? css("--ink") : css("--warn")) + "' stroke='" + css("--surface") + "' stroke-width='2'/>";
      }
      svg += "</svg>";
      var d = t.rank_lowvol - t["rank_" + h], v1 = t.vol["1h"];
      return "<tr data-sym='" + esc(t.symbol) + "'><td>" + t["rank_" + h] + "</td><td class='xs-tok'><i class='xs-dot' style='background:" + c + "'></i>" +
        esc(sh(t.symbol)) + "<span class='xs-sec'>" + esc(t.sector) + "</span></td><td class='xs-band'>" + svg + "</td><td>" + bp(q[1]) + "</td><td>[" + bp(a[0]) + ", " + bp(a[1]) + "]</td>" +
        "<td>" + t.rank_lowvol + "</td><td>" + (d > 0 ? "+" : "") + d + "</td><td>" + v1.v3[5].toFixed(2) + " / " + v1.har[5].toFixed(2) + "%</td></tr>";
    }).join("");
    $("xs-tbl").innerHTML = "<caption>Sorted by v3's " + h + " relative median (rank 1 = highest). Context only, not a signal: a high rank reads as a calm, " +
      "median-type outcome, not expected excess return (R-XS-2).</caption>" + head + "<tbody>" + rows + "</tbody>";
    Array.prototype.forEach.call($("xs-tbl").querySelectorAll("tbody tr"), function (tr) {
      var t = tok.filter(function (x) { return x.symbol === tr.getAttribute("data-sym"); })[0];
      tr.addEventListener("mousemove", function (ev) { showTip(ev, rowTip(t, live)); });
      tr.addEventListener("mouseleave", hideTip);
    });
  }

  function rowTip(t, live) {
    var h = X.h, r = t.rel[h], v = t.vol;
    var s = "<div class='t'>" + esc(sh(t.symbol)) + " (" + esc(t.sector) + ")</div><div class='m'>" +
      h + " rel median " + bp(r.q_bp[1], 1) + " bp; raw 80% [" + bp(r.q_bp[0]) + ", " + bp(r.q_bp[2]) + "]; ACI 80% [" + bp(r.aci_bp[0]) + ", " + bp(r.aci_bp[1]) + "] " +
      "(Q " + (fin(r.Q) ? r.Q.toFixed(3) : "-") + ", alpha_t " + (fin(r.alpha_t) ? r.alpha_t.toFixed(3) : "-") + ")<br>" +
      "v3 rank 4h " + t.rank_4h + ", 24h " + t.rank_24h + "; low-vol rank " + t.rank_lowvol + " (lsig2 " + t.lsig2.toFixed(2) + ")<br>" +
      "1h |r| v3 q50 " + v["1h"].v3[3].toFixed(2) + "%, q90 " + v["1h"].v3[5].toFixed(2) + "% | HAR q90 " + v["1h"].har[5].toFixed(2) + "%<br>" +
      "4h |r| v3 q90 " + v["4h"].v3[5].toFixed(2) + "% | HAR q90 " + v["4h"].har[5].toFixed(2) + "%<br>attention received " + (fin(t.att_received) ? t.att_received.toFixed(2) + "x uniform" : "-");
    if (live && fin(live.rel[t.symbol])) s += "<br>live move vs basket since " + X.line.forecast_hour.slice(11, 16) + ": " + bp(live.rel[t.symbol], 1) + " bp";
    if (h === "24h") s += "<br>24h ACI band known to under-cover (holdout 0.749 at 80%)";
    return s + "</div>";
  }

  function drawMap(L) {
    var svg = $("xs-map"); while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = 460, H = 330, pad = 30, h = X.h;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var pts = L.tokens.filter(function (t) { return t.present && t.xy && fin(t.xy[0]); });
    if (!pts.length) { svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "xs-muted" }, svg).textContent = "no projection"; return; }
    var an = (L.map && L.map.anchors) || {}, xs = [], ys = [];
    pts.forEach(function (t) { xs.push(t.xy[0]); ys.push(t.xy[1]); var a = an[t.symbol]; if (a) { xs.push(a[0]); ys.push(a[1]); } });
    var mnx = Math.min.apply(null, xs), mxx = Math.max.apply(null, xs), mny = Math.min.apply(null, ys), mxy = Math.max.apply(null, ys);
    var SX = function (v) { return pad + (v - mnx) / ((mxx - mnx) || 1) * (W - 2 * pad); }, SY = function (v) { return H - pad - (v - mny) / ((mxy - mny) || 1) * (H - 2 * pad); };
    var ev = (L.map && L.map.explained) || [];
    svgEl("text", { x: W - pad, y: H - 8, "text-anchor": "end", class: "xs-axl" }, svg).textContent = "PC1" + (ev[0] ? " (" + Math.round(100 * ev[0]) + "%)" : "") + " ->";
    svgEl("text", { x: 8, y: 14, class: "xs-axl" }, svg).textContent = "PC2" + (ev[1] ? " (" + Math.round(100 * ev[1]) + "%)" : "");
    var mxm = 0;
    pts.forEach(function (t) { var r = t.rel[h]; mxm = Math.max(mxm, Math.abs(r.q_bp[1] / ((r.q_bp[2] - r.q_bp[0]) / 2 || 1))); });
    pts.forEach(function (t) {
      var a = an[t.symbol], c = secColor(t.sector);
      if (a) {
        svgEl("line", { x1: SX(a[0]), y1: SY(a[1]), x2: SX(t.xy[0]), y2: SY(t.xy[1]), stroke: c, "stroke-opacity": 0.22, "stroke-width": 1 }, svg);
        svgEl("circle", { cx: SX(a[0]), cy: SY(a[1]), r: 2.5, fill: "none", stroke: c, "stroke-opacity": 0.5 }, svg);
      }
    });
    var boxes = [];   // greedy label placement: right, left, above, below of the dot, first free slot
    function free(b) { return boxes.every(function (o) { return b[0] > o[2] || b[2] < o[0] || b[1] > o[3] || b[3] < o[1]; }); }
    pts.slice().sort(function (p, q) { return p.xy[1] - q.xy[1]; }).forEach(function (t) {
      var x = SX(t.xy[0]), y = SY(t.xy[1]), w = 6.2 * sh(t.symbol).length, opts = [[x + 8, y + 4, "start"], [x - 8, y + 4, "end"], [x, y - 9, "middle"], [x, y + 16, "middle"]];
      var pick = opts[0];
      for (var i = 0; i < opts.length; i++) {
        var o = opts[i], x0 = o[2] === "start" ? o[0] : o[2] === "end" ? o[0] - w : o[0] - w / 2, b = [x0, o[1] - 9, x0 + w, o[1] + 2];
        if (free(b)) { pick = o; boxes.push(b); break; }
        if (i === opts.length - 1) boxes.push(b);
      }
      t._lbl = pick;
    });
    pts.forEach(function (t) {
      var c = secColor(t.sector), x = SX(t.xy[0]), y = SY(t.xy[1]);
      var r = t.rel[h], z = r.q_bp[1] / ((r.q_bp[2] - r.q_bp[0]) / 2 || 1), len = 26 * z / (mxm || 1);
      if (Math.abs(len) > 1.5) {
        svgEl("line", { x1: x, y1: y, x2: x, y2: y - len, stroke: css("--ink-2"), "stroke-width": 1.5, "marker-end": "url(#xs-arr)" }, svg);
      }
      var g = svgEl("g", { class: "xs-pt" }, svg);
      svgEl("circle", { cx: x, cy: y, r: 10, fill: "transparent" }, g);
      svgEl("circle", { cx: x, cy: y, r: 5, fill: c, stroke: css("--surface"), "stroke-width": 2 }, g);
      svgEl("text", { x: t._lbl[0], y: t._lbl[1], "text-anchor": t._lbl[2], class: "xs-lbl" }, g).textContent = sh(t.symbol);
      g.addEventListener("mousemove", function (e) { showTip(e, rowTip(t, liveRel())); });
      g.addEventListener("mouseleave", hideTip);
    });
    var defs = svgEl("defs", {}, svg), m = svgEl("marker", { id: "xs-arr", viewBox: "0 0 6 6", refX: 3, refY: 3, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
    svgEl("path", { d: "M0 0 L6 3 L0 6 z", fill: css("--ink-2") }, m);
  }

  function drawAttention(L) {
    var A = L.attention; if (!A) return;
    var hubs = A.hubs || [], mx = Math.max.apply(null, hubs.map(function (x) { return x[1]; }).concat([1]));
    $("xs-hubs").innerHTML = "<caption>Most attended keys (received from the other tokens, 1 = uniform)</caption><tbody>" + hubs.map(function (x) {
      return "<tr><td>" + esc(x[0] === "MARKET" ? "market" : sh(x[0])) + "</td><td class='xs-hubbar'><i style='width:" + (100 * x[1] / mx).toFixed(0) + "%'></i></td><td>" + x[1].toFixed(2) + "x</td></tr>";
    }).join("") + "<tr><td>mkt share</td><td></td><td>" + (fin(A.market_share) ? (100 * A.market_share).toFixed(1) + "%" : "-") + "</td></tr></tbody>";
    var svg = $("xs-heat"); while (svg.firstChild) svg.removeChild(svg.firstChild);
    var nq = A.queries.length, nk = A.keys.length, cs = 11, ox = 38, oy = 30, W = ox + nk * cs + 4, H = oy + nq * cs + 4;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var base = css("--v1"), uni = 1 / nk;
    A.keys.forEach(function (k, j) {
      var t = svgEl("text", { x: ox + j * cs + cs / 2, y: oy - 4, class: "xs-hl", transform: "rotate(-60 " + (ox + j * cs + cs / 2) + " " + (oy - 4) + ")" }, svg);
      t.textContent = k === "MARKET" ? "MKT" : sh(k);
    });
    A.queries.forEach(function (q, i) {
      svgEl("text", { x: ox - 3, y: oy + i * cs + cs - 2, "text-anchor": "end", class: "xs-hl" }, svg).textContent = sh(q);
      A.w[i].forEach(function (w, j) {
        var ratio = w / uni, op = Math.max(0.03, Math.min(1, Math.log(1 + ratio) / Math.log(1 + 4)));
        var r = svgEl("rect", { x: ox + j * cs, y: oy + i * cs, width: cs - 1, height: cs - 1, fill: base, "fill-opacity": op.toFixed(3) }, svg);
        r.addEventListener("mousemove", function (e) {
          showTip(e, "<div class='t'>" + esc(sh(q)) + " -> " + esc(A.keys[j] === "MARKET" ? "market token" : sh(A.keys[j])) + "</div><div class='m'>weight " + w.toFixed(3) +
            " (" + ratio.toFixed(2) + "x uniform; seeds and heads averaged). Descriptive only: removing attention did not hurt the forecasts (R-XS-4).</div>");
        });
        r.addEventListener("mouseleave", hideTip);
      });
    });
  }

  function drawFoot(L, live, el) {
    var sm = L.summary || {}, ru = L.running || {}, b = L.tokens.filter(function (t) { return t.symbol === "BTCUSDT" && t.present; })[0], v1 = L.vs_v1;
    function f(v, d) { return fin(v) ? (v >= 0 ? "+" : "") + (+v).toFixed(d === undefined ? 2 : d) : "n/a"; }
    var parts = ["How much is just low vol: rank corr of v3's median with -lsig2 " + f(sm.rank_corr_lowvol_4h) + " (4h) / " + f(sm.rank_corr_lowvol_24h) +
      " (24h); " + (sm.top5_overlap_lowvol) + " of v3's top 5 are in the low-vol top 5."];
    if (b) parts.push("BTC 1h |r|: v3 q50 " + b.vol["1h"].v3[3].toFixed(2) + "%, q90 " + b.vol["1h"].v3[5].toFixed(2) + "% vs log-HAR-t q90 " + b.vol["1h"].har[5].toFixed(2) + "%" +
      (v1 ? "; v1 90% half-width " + v1.half90_pct.toFixed(2) + "%" : "") + ".");
    var r4 = ru.rel_4h, r24 = ru.rel_24h, v = ru.vol_1h;
    parts.push("Running (scored lines): 4h IC v3 " + (r4 ? f(r4.ic_v3, 3) + " vs low-vol " + f(r4.ic_lowvol, 3) + " (n " + r4.n_lines + ")" : "n/a") +
      "; 24h " + (r24 ? f(r24.ic_v3, 3) + " vs " + f(r24.ic_lowvol, 3) + " (n " + r24.n_lines + ")" : "n/a") +
      "; 1h vol pinball v3 / HAR " + (v && fin(v.ratio) ? v.ratio.toFixed(3) + " (n " + v.n_lines + ")" : "n/a") + ".");
    if (X.h === "24h") parts.push("The 24h ACI band is known to under-cover (holdout 0.749 at 80%, 6/23 tokens within 3 pp).");
    parts.push(live ? "Live dot: move vs the equal-weight basket of " + live.n + " tokens since " + L.forecast_hour.slice(11, 16) +
      " UTC (Bybit tickers); dashed box = the ACI band scaled by sqrt(elapsed / " + X.h + ") = " + Math.sqrt(Math.max(0.02, Math.min(1, el))).toFixed(2) + "; amber = outside it." :
      "Live moves: waiting for the Bybit tickers feed.");
    parts.push("Map: fixed PCA projection of the network's per-token representation (fitted once on the training block); rings = training-mean position, " +
      "arrows = sign and size of the " + X.h + " relative median (median / band half-width).");
    $("xs-foot").innerHTML = parts.map(function (p) { return "<span>" + esc(p) + "</span>"; }).join(" ");
    $("xs-models").textContent = (L.models ? L.models.model_id + " " + L.models.version + " | spec " + L.models.spec_sha256 + " | projection " + L.models.projection_sha256 +
      " | inputs " + L.inputs_hash + " | data to " + L.data_max_ts.slice(0, 16) : "");
  }

  function boot() {
    if (!$("xs-panel")) return;
    Array.prototype.forEach.call(document.querySelectorAll("[data-xsh]"), function (b) {
      b.addEventListener("click", function () {
        X.h = b.getAttribute("data-xsh");
        document.querySelectorAll("[data-xsh]").forEach(function (x) { x.classList.toggle("on", x === b); });
        draw(true);
      });
    });
    load();
    setInterval(load, POLL_MS);
    setInterval(function () { draw(false); }, 15000);
    new MutationObserver(function () { draw(true); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(function () { draw(true); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
