/* BTC network monitor: live price (Bybit public websocket, in the browser), the frozen network's bands and activations
   (local read-only server), Claude's read. Research only: nothing here can trade. All times UTC. */
(function () {
  "use strict";

  var WS_URL = "wss://stream.bybit.com/v5/public/linear";
  var POLL_MS = 30000, PING_MS = 20000, WATCHDOG_MS = 45000, CHART_MIN_MS = 450;
  var HOUR = 3600000;
  var LEVELS = ["80", "90", "95"];
  var THINK_LABEL = { variance: "Variance vs the past week", latent: "Latent state", versions: "v1 vs v2", coverage: "Coverage alarms", field: "Field" };

  var S = {
    state: null, act: null, edges: {}, net: "1h", seed: 0, view: "1h",
    candles: [], last: null, lastMsg: 0, chg24: null,
    lineId: null, v2Key: null, userZoom: false, chartTimer: null, chartAt: 0, plotted: false, klineNote: ""
  };

  // ------------------------------------------------------------------ helpers
  function $(id) { return document.getElementById(id); }
  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function fin(v) { return v !== null && v !== undefined && isFinite(v); }
  function f(v, d, sign) { if (!fin(v)) return "-"; d = d === undefined ? 2 : d; var s = (+v).toFixed(d); return (sign !== false && v >= 0 ? "+" : "") + s; }
  function px(v) { return fin(v) ? (+v).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) : "-"; }
  function hhmm(t) { return new Date(typeof t === "number" ? t : Date.parse(t)).toISOString().slice(11, 16); }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function rgba(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map(function (c) { return c + c; }).join("");
    var n = parseInt(h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; });
    });
  }
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } return null; }

  // ------------------------------------------------------------------ theme
  var THEMES = ["auto", "light", "dark"];
  function applyTheme(t) {
    if (t === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    $("theme").textContent = "Theme: " + t;
    redrawAll();
  }
  $("theme").addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme") || "auto";
    var nxt = THEMES[(THEMES.indexOf(cur) + 1) % 3];
    store("btcmon.theme", nxt);
    applyTheme(nxt);
  });
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(function () { redrawAll(); });
  }
  function redrawAll() { scheduleChart(true); drawLatent(); drawRealised(); drawNet(false); drawMind(); }

  // ------------------------------------------------------------------ tooltip
  var tip = $("tip");
  function showTip(ev, html) {
    tip.innerHTML = html; tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight, x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip.hidden = true; }

  // ------------------------------------------------------------------ data: state
  function line() { return S.state && S.state.v1; }
  function v2line() { return S.state && S.state.v2; }

  function loadState() {
    return getJSON("/api/state").then(function (st) {
      S.state = st;
      $("m-poll").textContent = "updated " + new Date().toISOString().slice(11, 19);
      var L = st.v1;
      var v2k = st.v2 ? st.v2.id : null;
      var newLine = L && L.id !== S.lineId;
      if (L && (newLine || v2k !== S.v2Key)) {
        var first = S.lineId === null;
        S.lineId = L.id; S.v2Key = v2k;
        loadActivations(L.id, newLine || first);
      }
      renderHeader(); renderRegime(); renderClaude(); renderHlive(); renderFooter();
      drawLatent(); drawRealised(); scheduleChart(true); drawMind();
    }).catch(function (e) { $("m-poll").textContent = "state error: " + e.message; });
  }

  function loadActivations(id, fire) {
    return getJSON("/api/activations?id=" + encodeURIComponent(id)).then(function (a) {
      S.act = a;
      renderThinking();
      drawNet(fire);
      if (!S.edges["1h"]) getJSON("/api/edges?h=1h").then(function (e) { S.edges["1h"] = e; });
      if (!S.edges["24h"]) getJSON("/api/edges?h=24h").then(function (e) { S.edges["24h"] = e; });
    }).catch(function (e) { $("think").innerHTML = "<li>activations unavailable: " + esc(e.message) + "</li>"; });
  }

  // ------------------------------------------------------------------ data: klines + websocket
  function upsert(row) {
    var c = S.candles, n = c.length;
    if (n && c[n - 1][0] === row[0]) c[n - 1] = row;
    else if (!n || c[n - 1][0] < row[0]) c.push(row);
    else {
      var i = c.findIndex(function (x) { return x[0] === row[0]; });
      if (i >= 0) c[i] = row; else { c.push(row); c.sort(function (a, b) { return a[0] - b[0]; }); }
    }
    var cut = Date.now() - 48 * HOUR;
    while (c.length && c[0][0] < cut) c.shift();
  }
  function loadKlines() {
    return getJSON("/api/kline?hours=48").then(function (k) {
      k.rows.forEach(upsert);
      if (k.rows.length && S.last === null) { S.last = k.rows[k.rows.length - 1][4]; onTick(); }
      S.klineNote = "";
      scheduleChart(true); drawMind();
    }).catch(function (e) { S.klineNote = "History unavailable (" + e.message + "); building candles from the live feed."; scheduleChart(true); });
  }

  var ws = null, pingT = null, backoff = 1000, everConnected = false;
  function setFeed(cls, txt) { $("ws-dot").className = "dot " + cls; $("ws-txt").textContent = txt; }
  function connect() {
    setFeed("wait", "connecting");
    try { ws = new WebSocket(WS_URL); } catch (e) { setFeed("bad", "websocket blocked"); return setTimeout(connect, backoff); }
    ws.onopen = function () {
      ws.send(JSON.stringify({ op: "subscribe", args: ["tickers.BTCUSDT", "kline.1.BTCUSDT"] }));
      clearInterval(pingT);
      pingT = setInterval(function () { if (ws.readyState === 1) ws.send(JSON.stringify({ op: "ping" })); }, PING_MS);
      backoff = 1000; S.lastMsg = Date.now();
      setFeed("ok", "live");
      if (everConnected) loadKlines();  // fill any gap left by the outage
      everConnected = true;
    };
    ws.onmessage = function (e) {
      S.lastMsg = Date.now();
      var m; try { m = JSON.parse(e.data); } catch (x) { return; }
      if (m.topic === "tickers.BTCUSDT" && m.data) {
        if (m.data.lastPrice) { S.last = +m.data.lastPrice; }
        if (m.data.price24hPcnt) S.chg24 = 100 * +m.data.price24hPcnt;
        onTick();
      } else if (m.topic === "kline.1.BTCUSDT" && m.data) {
        m.data.forEach(function (k) { upsert([+k.start, +k.open, +k.high, +k.low, +k.close, +k.volume]); S.last = +k.close; });
        onTick(); scheduleChart(false);
      }
    };
    ws.onclose = function () {
      clearInterval(pingT); setFeed("bad", "reconnecting in " + Math.round(backoff / 1000) + " s");
      setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 30000);
    };
    ws.onerror = function () { try { ws.close(); } catch (x) { /* closed */ } };
  }
  setInterval(function () {  // watchdog: a silent socket is reconnected
    if (ws && ws.readyState === 1 && Date.now() - S.lastMsg > WATCHDOG_MS) { setFeed("wait", "silent, reconnecting"); ws.close(); }
  }, 5000);

  function onTick() {
    $("px").textContent = px(S.last);
    $("px-chg").textContent = fin(S.chg24) ? f(S.chg24, 2) + "% 24h" : "";
    drawMind();
  }

  // ------------------------------------------------------------------ header and clock
  function renderHeader() {
    var L = line(); if (!L) { $("m-line").textContent = "no line yet"; return; }
    var n1 = L.net["1h"];
    $("m-line").textContent = L.id + (L.flags.length ? " (" + L.flags.join(", ") + ")" : "");
    $("m-window").textContent = hhmm(n1.window_start) + "-" + hhmm(n1.window_end);
  }
  setInterval(function () {
    var L = line(); if (!L) return;
    var we = Date.parse(L.net["1h"].window_end), left = we - Date.now();
    if (left > 0) { var m = Math.floor(left / 60000), s = Math.floor((left % 60000) / 1000); $("m-left").textContent = m + ":" + String(s).padStart(2, "0"); }
    else $("m-left").textContent = "closed, next line ~" + hhmm(we + 3 * 60000);
    drawMind();
  }, 1000);

  // ------------------------------------------------------------------ cones
  function cone(t0, dur, lo, hi, P0, n) {
    n = n || 24; var xs = [], ys = [], i, fr;
    for (i = 0; i <= n; i++) { fr = i / n; xs.push(t0 + fr * dur); ys.push(P0 * Math.exp(hi / 100 * Math.sqrt(fr))); }
    for (i = n; i >= 0; i--) { fr = i / n; xs.push(t0 + fr * dur); ys.push(P0 * Math.exp(lo / 100 * Math.sqrt(fr))); }
    return { x: xs, y: ys };
  }
  function path(t0, dur, v, P0, n) {
    n = n || 24; var xs = [], ys = [];
    for (var i = 0; i <= n; i++) { var fr = i / n; xs.push(t0 + fr * dur); ys.push(P0 * Math.exp(v / 100 * Math.sqrt(fr))); }
    return { x: xs, y: ys };
  }
  function bandLines(t0, dur, lo, hi, P0) {
    var a = path(t0, dur, hi, P0), b = path(t0, dur, lo, P0);
    return { x: a.x.concat([null], b.x), y: a.y.concat([null], b.y) };
  }

  // ------------------------------------------------------------------ price chart
  function scheduleChart(now) {
    if (S.chartTimer) return;
    var wait = now ? 0 : Math.max(0, CHART_MIN_MS - (Date.now() - S.chartAt));
    S.chartTimer = setTimeout(function () { S.chartTimer = null; S.chartAt = Date.now(); drawChart(); }, wait);
  }

  function legendHTML(items) {
    $("price-legend").innerHTML = items.map(function (it) {
      return '<li><i style="' + it.style + '"></i>' + esc(it.label) + "</li>";
    }).join("");
  }

  function drawChart() {
    if (!window.Plotly) return;
    var gd = $("chart"), L = line(), c = S.candles;
    var C = { v1: css("--v1"), v2: css("--v2"), cl: css("--claude"), field: css("--field"), up: css("--up"), dn: css("--down"),
              ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"), grid: css("--grid"), surf: css("--surface") };
    var tr = [], shapes = [], ann = [], leg = [];
    tr.push({ type: "candlestick", name: "BTCUSDT 1m", x: c.map(function (r) { return r[0]; }), open: c.map(function (r) { return r[1]; }),
      high: c.map(function (r) { return r[2]; }), low: c.map(function (r) { return r[3]; }), close: c.map(function (r) { return r[4]; }),
      increasing: { line: { color: C.up, width: 1 }, fillcolor: C.up }, decreasing: { line: { color: C.dn, width: 1 }, fillcolor: C.dn },
      whiskerwidth: 0, hoverinfo: "x+y", showlegend: false });
    var yr = [Infinity, -Infinity];
    function grow(ys) { ys.forEach(function (y) { if (fin(y)) { yr[0] = Math.min(yr[0], y); yr[1] = Math.max(yr[1], y); } }); }
    var xr;
    if (L) {
      var n1 = L.net["1h"], n24 = L.net["24h"], P0 = L.close, t0 = Date.parse(n1.window_start);
      var ops = { "95": 0.09, "90": 0.13, "80": 0.2 };
      ["95", "90", "80"].forEach(function (k) {
        var b = n1.aci[k]; if (!fin(b.lo_pct)) return;
        var g = cone(t0, HOUR, b.lo_pct, b.hi_pct, P0);
        tr.push({ type: "scatter", mode: "lines", x: g.x, y: g.y, fill: "toself", fillcolor: rgba(C.v1, ops[k]), line: { width: 0 },
          hoverinfo: "skip", name: "v1 ACI " + k + "%" });
        if (S.view !== "24h") grow([g.y[24], g.y[25]]);
      });
      var md = path(t0, HOUR, n1.quantiles_pct.q50, P0);
      tr.push({ type: "scatter", mode: "lines", x: md.x, y: md.y, line: { color: C.v1, width: 1, dash: "dot" }, hoverinfo: "skip", name: "v1 median" });
      leg.push({ style: "background:" + rgba(C.v1, 0.35), label: "Network v1, ACI 80/90/95% (this hour, nested)" });
      if (n24 && n24.aci) {
        ["90", "80"].forEach(function (k) {
          var b = n24.aci[k]; if (!b || !fin(b.lo_pct)) return;
          var g = cone(t0, 24 * HOUR, b.lo_pct, b.hi_pct, P0, 48);
          tr.push({ type: "scatter", mode: "lines", x: g.x, y: g.y, fill: "toself", fillcolor: rgba(C.v1, k === "90" ? 0.035 : 0.05),
            line: { width: 0.6, color: rgba(C.v1, 0.25) }, hoverinfo: "skip", name: "24h ACI " + k + "%" });
          if (S.view === "24h") grow(g.y);
        });
        leg.push({ style: "background:" + rgba(C.v1, 0.08) + ";border:1px solid " + rgba(C.v1, 0.3), label: "24h net, ACI 80/90%" });
      }
      var v2 = v2line();
      if (v2 && v2.net["1h"].band) {
        ["80", "95"].forEach(function (k) {
          var b = v2.net["1h"].band[k]; if (!fin(b.lo_pct)) return;
          var g = bandLines(t0, HOUR, b.lo_pct, b.hi_pct, P0);
          tr.push({ type: "scatter", mode: "lines", x: g.x, y: g.y, line: { color: C.v2, width: k === "80" ? 1.6 : 1, dash: k === "80" ? "dash" : "dot" },
            hoverinfo: "skip", name: "v2 " + k + "%", connectgaps: false });
          if (S.view !== "24h") grow(g.y);
        });
        leg.push({ style: "border-top:2px dashed " + C.v2 + ";height:0", label: "v2 issued 80% (dash) / 95% (dot)" });
      }
      var cl = S.state.claude && S.state.claude.current_1h;
      if (cl) {
        var g2 = bandLines(t0, HOUR, cl.q10, cl.q90, P0);
        tr.push({ type: "scatter", mode: "lines", x: g2.x, y: g2.y, line: { color: C.cl, width: 1.8, dash: "dashdot" }, hoverinfo: "skip", name: "Claude 80%" });
        var m2 = path(t0, HOUR, cl.q50, P0);
        tr.push({ type: "scatter", mode: "lines", x: m2.x, y: m2.y, line: { color: C.cl, width: 1, dash: "dot" }, hoverinfo: "skip", name: "Claude median" });
        if (S.view !== "24h") grow(g2.y);
        leg.push({ style: "border-top:2px dashed " + C.cl + ";height:0", label: "Claude 80% (logged " + hhmm(cl.logged_at) + ")" });
      }
      // field touch levels
      var fh = L.field && L.field.horizons, fc = (L.field && L.field.close) || P0;
      if (fh && fh["1h"]) {
        [1, 2, 3].forEach(function (m) {
          [1, -1].forEach(function (sg) {
            var key = (sg > 0 ? "+" : "-") + m + "%", y = fc * (1 + sg * m / 100);
            shapes.push({ type: "line", xref: "x", yref: "y", x0: t0, x1: t0 + HOUR, y0: y, y1: y, line: { color: C.field, width: 1, dash: "dot" } });
            ann.push({ x: t0 + HOUR, y: y, xref: "x", yref: "y", xanchor: "right", yanchor: "bottom", showarrow: false,
              font: { size: 10, color: C.ink2, family: "IBM Plex Mono" }, bgcolor: rgba(C.surf.length === 7 ? C.surf : "#ffffff", 0.7),
              text: key + " " + Math.round(100 * fh["1h"].p_touch[key]) + "%" + (fh["4h"] ? " | 4h " + Math.round(100 * fh["4h"].p_touch[key]) + "%" : "") });
          });
        });
        leg.push({ style: "border-top:2px dotted " + C.field + ";height:0", label: "Field touch levels, P(touch) this hour / 4h" });
      }
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: t0, x1: t0, y0: 0, y1: 1, line: { color: C.ink3, width: 1, dash: "dash" } });
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: t0 + HOUR, x1: t0 + HOUR, y0: 0, y1: 1, line: { color: C.grid, width: 1 } });
      shapes.push({ type: "line", xref: "x", yref: "y", x0: t0 - 20 * 60000, x1: t0, y0: P0, y1: P0, line: { color: C.ink3, width: 1 } });
      ann.push({ x: t0, y: 1, xref: "x", yref: "paper", yanchor: "bottom", xanchor: "left", showarrow: false,
        font: { size: 10, color: C.ink3, family: "IBM Plex Mono" }, text: " " + L.id + " anchor " + px(P0) });
      var now = Date.now();
      if (S.view === "1h") xr = [t0 - 6 * HOUR, Math.max(t0 + HOUR, now) + 12 * 60000];
      else if (S.view === "24h") xr = [t0 - 12 * HOUR, t0 + 24 * HOUR];
      else xr = [c.length ? c[0][0] : t0 - 48 * HOUR, Math.max(t0 + HOUR, now) + 12 * 60000];
    } else {
      xr = [c.length ? c[0][0] : Date.now() - 6 * HOUR, Date.now() + 10 * 60000];
    }
    c.forEach(function (r) { if (r[0] >= xr[0] && r[0] <= xr[1]) { yr[0] = Math.min(yr[0], r[3]); yr[1] = Math.max(yr[1], r[2]); } });
    if (fin(S.last)) {
      grow([S.last]);
      shapes.push({ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: S.last, y1: S.last, line: { color: C.ink, width: 0.6, dash: "dot" } });
    }
    var pad = (yr[1] - yr[0]) * 0.06 || 50;
    var layout = {
      paper_bgcolor: C.surf, plot_bgcolor: C.surf, font: { family: "IBM Plex Mono, ui-monospace, monospace", size: 11, color: C.ink2 },
      margin: { l: 8, r: 64, t: 18, b: 28 }, showlegend: false, dragmode: "pan", hovermode: "x",
      xaxis: { type: "date", rangeslider: { visible: false }, gridcolor: C.grid, linecolor: C.grid, tickformat: "%H:%M\n%b %d", hoverformat: "%H:%M UTC" },
      yaxis: { side: "right", gridcolor: C.grid, tickformat: ",.0f", zeroline: false, fixedrange: false },
      shapes: shapes, annotations: ann, uirevision: "keep"
    };
    if (S.userZoom && gd.layout && gd.layout.xaxis) { layout.xaxis.range = gd.layout.xaxis.range; layout.yaxis.range = gd.layout.yaxis.range; }
    else { layout.xaxis.range = xr; if (fin(yr[0]) && fin(yr[1])) layout.yaxis.range = [yr[0] - pad, yr[1] + pad]; }
    Plotly.react(gd, tr, layout, { displaylogo: false, responsive: true, scrollZoom: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"] });
    if (!S.plotted) {
      S.plotted = true;
      gd.on("plotly_relayout", function (ev) {
        if (ev["xaxis.autorange"] || ev.autosize) { S.userZoom = false; scheduleChart(true); }
        else if (ev["xaxis.range[0]"] !== undefined || ev["xaxis.range"] || ev["yaxis.range[0]"] !== undefined) S.userZoom = true;
      });
    }
    legendHTML(leg);
    $("chart-note").textContent = (S.klineNote ? S.klineNote + " " : "") + (S.userZoom ? "Zoomed by hand; double-click the chart or pick a view to reset." : "");
  }
  Array.prototype.forEach.call(document.querySelectorAll("[data-view]"), function (b) {
    b.addEventListener("click", function () {
      S.view = b.getAttribute("data-view"); S.userZoom = false;
      document.querySelectorAll("[data-view]").forEach(function (x) { x.classList.toggle("on", x === b); });
      scheduleChart(true);
    });
  });

  // ------------------------------------------------------------------ where price is in the network's mind
  function closeAt(ms) {
    for (var i = S.candles.length - 1; i >= 0; i--) if (S.candles[i][0] === ms - 60000) return S.candles[i][4];
    return null;
  }
  function liveMove() {
    var L = line(); if (!L || !fin(S.last) || !fin(L.close)) return null;
    var n1 = L.net["1h"], ws = Date.parse(n1.window_start), we = Date.parse(n1.window_end), now = Date.now();
    var closed = now >= we, price = closed ? (closeAt(we) || S.last) : S.last;
    var mv = 100 * Math.log(price / L.close);
    var p = PCT.percentile(mv, n1.quantiles_pct), fr = clamp((Math.min(now, we) - ws) / (we - ws), 0, 1);
    var ps = fr > 0.02 ? PCT.percentile(mv, PCT.scaled(n1.quantiles_pct, fr)) : null;
    return { L: L, n1: n1, price: price, move: mv, p: p, ps: ps, frac: fr, closed: closed, z: mv / n1.sigma_pct };
  }
  var GZ = 2.6;  // gauge spans probit -2.6 .. +2.6 (percentile 0.5 .. 99.5)
  function gAngle(z) { return Math.PI * (1 - (clamp(z, -GZ, GZ) + GZ) / (2 * GZ)); }
  function arc(cx, cy, r, a0, a1) {
    var x0 = cx + r * Math.cos(a0), y0 = cy - r * Math.sin(a0), x1 = cx + r * Math.cos(a1), y1 = cy - r * Math.sin(a1);
    return "M" + x0 + " " + y0 + " A" + r + " " + r + " 0 0 1 " + x1 + " " + y1;
  }
  function drawMind() {
    var g = $("gauge"), M = liveMove();
    g.innerHTML = "";
    var cx = 160, cy = 150, R = 118;
    var segs = [[0.005, 0.05, "--neg", 0.9], [0.05, 0.25, "--neg", 0.35], [0.25, 0.75, "--neutral", 1], [0.75, 0.95, "--pos", 0.35], [0.95, 0.995, "--pos", 0.9]];
    segs.forEach(function (s) {
      svgEl("path", { d: arc(cx, cy, R, gAngle(zOf(s[0])), gAngle(zOf(s[1]))),
        style: "stroke:var(" + s[2] + ");stroke-opacity:" + s[3], "stroke-width": 16, fill: "none" }, g);
    });
    [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].forEach(function (p) {
      var a = gAngle(zOf(p)), r0 = R + 10, r1 = R + 16;
      svgEl("line", { x1: cx + r0 * Math.cos(a), y1: cy - r0 * Math.sin(a), x2: cx + r1 * Math.cos(a), y2: cy - r1 * Math.sin(a), style: "stroke:var(--ink-3)" }, g);
      var t = svgEl("text", { x: cx + (R + 26) * Math.cos(a), y: cy - (R + 26) * Math.sin(a) + 4, "text-anchor": "middle",
        style: "fill:var(--ink-3);font:10px var(--mono)" }, g);
      t.textContent = Math.round(100 * p);
    });
    if (!M) {
      var tt = svgEl("text", { x: cx, y: cy - 20, "text-anchor": "middle", style: "fill:var(--ink-3);font:13px var(--sans)" }, g);
      tt.textContent = "waiting for price and line";
      return;
    }
    var a = gAngle(zOf(M.p));
    if (M.ps !== null) {
      var as = gAngle(zOf(M.ps)), rr = R + 1;
      svgEl("circle", { cx: cx + rr * Math.cos(as), cy: cy - rr * Math.sin(as), r: 4, style: "fill:none;stroke:var(--ink-2);stroke-width:1.5" }, g);
    }
    var r0 = R - 34, r1 = R + 9, ap = 0.06;
    svgEl("path", { d: "M" + (cx + r0 * Math.cos(a)) + " " + (cy - r0 * Math.sin(a)) + " L" + (cx + r1 * Math.cos(a + ap * 0.35)) + " " + (cy - r1 * Math.sin(a + ap * 0.35)) +
      " L" + (cx + r1 * Math.cos(a - ap * 0.35)) + " " + (cy - r1 * Math.sin(a - ap * 0.35)) + " Z", style: "fill:var(--ink);stroke:var(--surface);stroke-width:1.5" }, g);
    var big = svgEl("text", { x: cx, y: cy - 22, "text-anchor": "middle", style: "fill:var(--ink);font:600 34px var(--mono)" }, g);
    big.textContent = PCT.ordinal(clamp(Math.round(100 * M.p), 1, 99));
    var sub = svgEl("text", { x: cx, y: cy - 4, "text-anchor": "middle", style: "fill:var(--ink-3);font:11px var(--sans);letter-spacing:.08em" }, g);
    sub.textContent = M.closed ? "PERCENTILE, HOUR CLOSED" : "PERCENTILE OF THE HOUR";
    $("mind-sentence").textContent = PCT.sentence(M.p) + (M.closed ? " (The hour has closed; this is its final position.)" : "");
    var aci = M.n1.aci, where = "outside the 95% band";
    for (var i = 0; i < LEVELS.length; i++) { var b = aci[LEVELS[i]]; if (M.move >= b.lo_pct && M.move <= b.hi_pct) { where = "inside the " + LEVELS[i] + "% band"; break; } }
    var v2 = v2line(), w2 = "";
    if (v2 && v2.net["1h"].band) {
      w2 = "outside 95%";
      for (var j = 0; j < LEVELS.length; j++) { var b2 = v2.net["1h"].band[LEVELS[j]]; if (M.move >= b2.lo_pct && M.move <= b2.hi_pct) { w2 = "inside " + LEVELS[j] + "%"; break; } }
    }
    var mins = Math.round(M.frac * 60);
    var rows = [
      ["Move since " + hhmm(M.n1.window_start), f(M.move, 3) + "%"],
      ["Price / anchor", px(M.price) + " / " + px(M.L.close)],
      ["In network sigmas", f(M.z, 2) + " (sigma " + f(M.n1.sigma_pct, 2, false) + "%)"],
      ["v1 ACI", where],
      ["Elapsed", mins + " of 60 min"],
      ["sqrt-time percentile", M.ps === null ? "-" : PCT.ordinal(clamp(Math.round(100 * M.ps), 1, 99)) + " (hollow dot)"]
    ];
    if (w2) rows.splice(4, 0, ["v2 issued", w2]);
    $("mind-kv").innerHTML = rows.map(function (r) { return "<dt>" + esc(r[0]) + "</dt><dd>" + esc(r[1]) + "</dd>"; }).join("");
    drawLadder(M);
    drawTouch(M);
  }
  function zOf(p) {  // probit, for gauge geometry only (Acklam's rational approximation)
    var a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    var b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    var c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    var d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    p = clamp(p, 1e-9, 1 - 1e-9);
    var q, r;
    if (p < 0.02425) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  function drawLadder(M) {
    var g = $("ladder"); g.innerHTML = "";
    var n1 = M.n1, v2 = v2line(), cl = S.state.claude && S.state.claude.current_1h;
    var rows = [{ k: "v1 ACI", bands: LEVELS.map(function (L) { return [L, n1.aci[L].lo_pct, n1.aci[L].hi_pct]; }), col: "--v1" }];
    if (v2 && v2.net["1h"].band) rows.push({ k: "v2 issued", bands: LEVELS.map(function (L) { return [L, v2.net["1h"].band[L].lo_pct, v2.net["1h"].band[L].hi_pct]; }), col: "--v2" });
    if (cl) rows.push({ k: "Claude 80%", bands: [["80", cl.q10, cl.q90]], col: "--claude" });
    var ext = Math.abs(M.move);
    rows.forEach(function (r) { r.bands.forEach(function (b) { ext = Math.max(ext, Math.abs(b[1]), Math.abs(b[2])); }); });
    ext = Math.max(0.5, ext * 1.08);
    var x0 = 74, x1 = 312, X = function (v) { return x0 + (v + ext) / (2 * ext) * (x1 - x0); };
    var y = 16, op = { "95": 0.25, "90": 0.45, "80": 0.8 };
    var step = ext > 3 ? 1 : 0.5;
    for (var t = -Math.floor(ext / step) * step; t <= ext + 1e-9; t += step) {
      svgEl("line", { x1: X(t), x2: X(t), y1: 8, y2: 132, style: "stroke:var(--grid)" }, g);
      var tx = svgEl("text", { x: X(t), y: 146, "text-anchor": "middle", style: "fill:var(--ink-3);font:10px var(--mono)" }, g);
      tx.textContent = (t > 0 ? "+" : "") + t.toFixed(1);
    }
    rows.forEach(function (r) {
      var lb = svgEl("text", { x: 0, y: y + 9, style: "fill:var(--ink-2);font:11px var(--sans)" }, g); lb.textContent = r.k;
      r.bands.slice().reverse().forEach(function (b) {
        if (!fin(b[1])) return;
        svgEl("rect", { x: X(b[1]), y: y, width: Math.max(1, X(b[2]) - X(b[1])), height: 12, rx: 2, style: "fill:var(" + r.col + ");fill-opacity:" + op[b[0]] }, g);
      });
      y += 22;
    });
    var lb2 = svgEl("text", { x: 0, y: y + 9, style: "fill:var(--ink-2);font:11px var(--sans)" }, g); lb2.textContent = "v1 quantiles";
    ["q05", "q10", "q25", "q50", "q75", "q90", "q95"].forEach(function (k) {
      var v = n1.quantiles_pct[k];
      svgEl("line", { x1: X(v), x2: X(v), y1: y, y2: y + 12, style: "stroke:var(--ink-2);stroke-width:" + (k === "q50" ? 2 : 1) }, g);
    });
    svgEl("line", { x1: X(M.move), x2: X(M.move), y1: 6, y2: y + 16, style: "stroke:var(--ink);stroke-width:2" }, g);
    var mt = svgEl("text", { x: X(M.move), y: y + 28 > 132 ? 132 : y + 28, "text-anchor": X(M.move) > 260 ? "end" : "start",
      style: "fill:var(--ink);font:600 11px var(--mono)" }, g);
    mt.textContent = " now " + f(M.move, 2) + "%";
    g.setAttribute("aria-label", "Bands for this hour in % log return; live move " + f(M.move, 2) + "%");
  }

  function drawTouch(M) {
    var L = M.L, fh = L.field && L.field.horizons, t = $("touch-tbl");
    if (!fh || !fh["1h"]) { t.innerHTML = ""; return; }
    var fc = L.field.close || L.close, t0 = Date.parse(M.n1.window_start), hi = -Infinity, lo = Infinity;
    S.candles.forEach(function (r) { if (r[0] >= t0 && r[0] < t0 + HOUR) { hi = Math.max(hi, r[2]); lo = Math.min(lo, r[3]); } });
    var rows = [];
    [3, 2, 1].forEach(function (m) { rows.push(["+" + m + "%", fc * (1 + m / 100), 1]); });
    [1, 2, 3].forEach(function (m) { rows.push(["-" + m + "%", fc * (1 - m / 100), -1]); });
    t.innerHTML = "<caption>Field (" + esc(L.field.cell || "") + ") touch probabilities from " + px(fc) + "</caption><tr><th>Level</th><th>Price</th><th>1h</th><th>4h</th><th>This hour</th></tr>" +
      rows.map(function (r) {
        var hit = r[2] > 0 ? hi >= r[1] : lo <= r[1];
        return "<tr><td>" + r[0] + "</td><td>" + px(r[1]) + "</td><td>" + Math.round(100 * fh["1h"].p_touch[r[0]]) + "%</td><td>" +
          (fh["4h"] ? Math.round(100 * fh["4h"].p_touch[r[0]]) + "%" : "-") + "</td><td>" + (hit ? "touched" : "-") + "</td></tr>";
      }).join("");
  }

  // ------------------------------------------------------------------ network firing
  $("refire").addEventListener("click", function () { drawNet(true); });
  Array.prototype.forEach.call(document.querySelectorAll("[data-net]"), function (b) {
    b.addEventListener("click", function () {
      S.net = b.getAttribute("data-net");
      document.querySelectorAll("[data-net]").forEach(function (x) { x.classList.toggle("on", x === b); });
      drawNet(true);
    });
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-seed]"), function (b) {
    b.addEventListener("click", function () {
      S.seed = +b.getAttribute("data-seed");
      document.querySelectorAll("[data-seed]").forEach(function (x) { x.classList.toggle("on", x === b); });
      drawNet(true);
    });
  });

  function sparkline(xs, ys, xnow, w, h) {
    if (!xs || !ys) return "";
    var mnx = xs[0], mxx = xs[xs.length - 1], mny = Math.min.apply(null, ys.concat([0])), mxy = Math.max.apply(null, ys.concat([0]));
    var X = function (v) { return 4 + (v - mnx) / (mxx - mnx || 1) * (w - 8); }, Y = function (v) { return h - 4 - (v - mny) / (mxy - mny || 1) * (h - 8); };
    var d = xs.map(function (x, i) { return (i ? "L" : "M") + X(x).toFixed(1) + " " + Y(ys[i]).toFixed(1); }).join("");
    var s = '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + " " + h + '">' +
      '<line x1="4" x2="' + (w - 4) + '" y1="' + Y(0) + '" y2="' + Y(0) + '" style="stroke:var(--grid)"/>' +
      '<path d="' + d + '" style="fill:none;stroke:var(--ink-2);stroke-width:1.5"/>';
    if (fin(xnow)) {
      var xc = clamp(xnow, mnx, mxx), k = 0;
      for (var i = 0; i < xs.length - 1; i++) if (xs[i] <= xc) k = i;
      var fr = (xc - xs[k]) / ((xs[k + 1] - xs[k]) || 1), yv = ys[k] + fr * ((ys[k + 1] || ys[k]) - ys[k]);
      s += '<line x1="' + X(xc) + '" x2="' + X(xc) + '" y1="2" y2="' + (h - 2) + '" style="stroke:var(--ink-3);stroke-dasharray:2 2"/>' +
        '<circle cx="' + X(xc) + '" cy="' + Y(yv) + '" r="3.5" style="fill:var(--ink)"/>';
    }
    return s + "</svg>";
  }

  function drawNet(fire) {
    var svg = $("net"), A = S.act && S.act.horizons && S.act.horizons[S.net];
    svg.innerHTML = "";
    if (!A) { svgEl("text", { x: 20, y: 30, class: "lbl" }, svg).textContent = "Waiting for activations."; return; }
    var seed = A.seeds[Math.min(S.seed, A.seeds.length - 1)];
    var W = Math.max(620, svg.clientWidth || 820), H = 560;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var nIn = A.features.length, nLat = seed.latent.length, nOut = 7;
    var xIn = 190, xLat = Math.round(W * 0.53), xOut = W - 170;
    var yIn = function (i) { return 44 + i * (H - 64) / (nIn - 1); };
    var yLat = function (j) { return H / 2 - (nLat - 1) * 40 + j * 80; };
    var yOut = function (k) { return 70 + (nOut - 1 - k) * (H - 120) / (nOut - 1); };  // q95 on top
    [["Inputs (standardised)", 8, "start"], ["Latent z", xLat, "middle"], ["Quantile head", xOut, "middle"]].forEach(function (h) {
      svgEl("text", { x: h[1], y: 16, class: "hd", "text-anchor": h[2] }, svg).textContent = h[0];
    });
    var gE = svgEl("g", {}, svg), gS = svgEl("g", {}, svg), gH = svgEl("g", {}, svg), gN = svgEl("g", {}, svg);
    var E1 = seed.enc_edges, E2 = seed.head_edges;
    var m1 = 0, m2 = 0, i, j, k;
    for (j = 0; j < nLat; j++) for (i = 0; i < nIn; i++) m1 = Math.max(m1, Math.abs(E1[j][i]));
    for (k = 0; k < nOut; k++) for (j = 0; j < nLat; j++) m2 = Math.max(m2, Math.abs(E2[k][j]));
    var cat = S.edges[S.net] && S.edges[S.net].seeds[Math.min(S.seed, S.edges[S.net].seeds.length - 1)];
    var latNames = seed.latent.map(function (_, q) { return "z" + (q + 1); });
    var qn = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"];
    var list = [];
    for (j = 0; j < nLat; j++) for (i = 0; i < nIn; i++) list.push({ L: 1, i: i, j: j, v: E1[j][i], m: m1, x1: xIn, y1: yIn(i), x2: xLat, y2: yLat(j) });
    for (k = 0; k < nOut; k++) for (j = 0; j < nLat; j++) list.push({ L: 2, i: j, j: k, v: E2[k][j], m: m2, x1: xLat, y1: yLat(j), x2: xOut, y2: yOut(k) });
    list.sort(function (a, b) { return Math.abs(a.v) - Math.abs(b.v); });
    list.forEach(function (e) {
      var s = Math.abs(e.v) / (e.m || 1), mx = (e.x1 + e.x2) / 2;
      var d = "M" + e.x1 + " " + e.y1 + " C" + mx + " " + e.y1 + " " + mx + " " + e.y2 + " " + e.x2 + " " + e.y2;
      var col = e.v >= 0 ? "var(--pos)" : "var(--neg)", wdt = 0.4 + 5.6 * s, op = 0.12 + 0.88 * Math.pow(s, 0.7);
      svgEl("path", { d: d, class: "edge", style: "stroke:" + col + ";stroke-width:" + wdt.toFixed(2) + ";stroke-opacity:" + op.toFixed(3) }, gE);
      svgEl("path", { d: d, class: "spark l" + e.L, pathLength: 1, "stroke-dasharray": "0.14 1.2",
        style: "stroke-width:" + (wdt + 1.2).toFixed(2) + ";stroke-opacity:" + (0.25 + 0.75 * s).toFixed(2) }, gS);
      var hit = svgEl("path", { d: d, class: "hit" }, gH);
      hit.addEventListener("mousemove", function (ev) { showTip(ev, edgeTip(e, A, seed, cat, latNames, qn)); });
      hit.addEventListener("mouseleave", hideTip);
    });
    // input nodes
    var att = A.attribution;
    for (i = 0; i < nIn; i++) {
      var zv = A.z[i], zr = A.zraw[i], miss = A.missing[i], y = yIn(i);
      var fillc = miss ? "var(--neutral)" : (zv >= 0 ? "var(--pos)" : "var(--neg)");
      var c = svgEl("circle", { cx: xIn, cy: y, r: 7, class: "node" + (fin(zr) && Math.abs(zr) > 4 ? " alarm" : ""),
        style: "fill:" + fillc + ";fill-opacity:" + (miss ? 0.6 : (0.15 + 0.85 * Math.min(1, Math.abs(zv) / 3)).toFixed(2)) }, gN);
      svgEl("circle", { cx: xIn, cy: y, r: 7, class: "pulse l0" }, gN);
      svgEl("text", { x: 8, y: y + 4, class: "lbl" }, gN).textContent = A.features[i];
      svgEl("text", { x: xIn - 14, y: y + 4, "text-anchor": "end" }, gN).textContent = miss ? "n/a" : f(zv, 2);
      (function (i) { c.addEventListener("mousemove", function (ev) { showTip(ev, inputTip(A, i)); }); c.addEventListener("mouseleave", hideTip); })(i);
    }
    // latent nodes
    var lat = A.latent;
    for (j = 0; j < nLat; j++) {
      var sd = (seed.latent[j] - lat.zmu[j]) / lat.zsd[j], yl = yLat(j);
      var cl = svgEl("circle", { cx: xLat, cy: yl, r: 13, class: "node", style: "fill:" + (sd >= 0 ? "var(--pos)" : "var(--neg)") + ";fill-opacity:" + (0.2 + 0.8 * Math.min(1, Math.abs(sd) / 2.5)).toFixed(2) }, gN);
      svgEl("circle", { cx: xLat, cy: yl, r: 13, class: "pulse l1" }, gN);
      svgEl("text", { x: xLat, y: yl - 19, "text-anchor": "middle", class: "lbl" }, gN).textContent = latNames[j] + " " + f(seed.latent[j], 2);
      var nxt = lat.zhat_next && S.seed === 0 ? " next " + f((lat.zhat_next[j] - lat.zmu[j]) / lat.zsd[j], 2) : "";
      svgEl("text", { x: xLat, y: yl + 28, "text-anchor": "middle" }, gN).textContent = f(sd, 2) + " sd" + nxt;
      (function (j) { cl.addEventListener("mousemove", function (ev) { showTip(ev, latentTip(A, seed, j)); }); cl.addEventListener("mouseleave", hideTip); })(j);
    }
    // outputs
    for (k = 0; k < nOut; k++) {
      var yo = yOut(k), qv = A.quantiles_pct[k];
      var co = svgEl("circle", { cx: xOut, cy: yo, r: 9, class: "node", style: "fill:var(--ink-2);fill-opacity:" + (k === 3 ? 0.9 : 0.45) }, gN);
      svgEl("circle", { cx: xOut, cy: yo, r: 9, class: "pulse l2" }, gN);
      svgEl("text", { x: xOut + 16, y: yo + 4, class: "lbl" }, gN).textContent = qn[k];
      svgEl("text", { x: xOut + 50, y: yo + 4 }, gN).textContent = f(qv, 2) + "%";
      (function (k) { co.addEventListener("mousemove", function (ev) { showTip(ev, outTip(A, seed, k, qn)); }); co.addEventListener("mouseleave", hideTip); })(k);
    }
    var sig = A.sigma_pct;
    svgEl("text", { x: xOut + 16, y: H - 18 }, svg).textContent = "sigma " + f(sig, 2, false) + "%";
    svgEl("text", { x: xOut + 16, y: H - 4, style: "fill:var(--ink-3)" }, svg).textContent = "scale " + f(A.scale_pct, 2, false) + "%";
    var dif = A.max_abs_diff_vs_stored_pct;
    $("net-check").textContent = A.h + " net, refit " + A.refit_month + " (" + A.pt_sha256 + "): forward pass vs stored quantiles, max |diff| " +
      (fin(dif) ? dif.toExponential(1) : "n/a") + " pp; 2-seed mean shown on outputs";
    if (fire) {
      svg.classList.remove("firing"); void svg.getBoundingClientRect(); svg.classList.add("firing");
      setTimeout(function () { svg.classList.remove("firing"); }, 2200);
    }
  }

  function edgeTip(e, A, seed, cat, latNames, qn) {
    var layer = e.L === 1 ? "enc" : "head", src = e.L === 1 ? A.features[e.i] : latNames[e.i], dst = e.L === 1 ? latNames[e.j] : "o" + e.j + " (" + qn[e.j] + ")";
    var xin = e.L === 1 ? A.z[e.i] : seed.latent[e.i];
    var tot = 0, row = e.L === 1 ? seed.enc_edges[e.j] : seed.head_edges[e.j];
    row.forEach(function (v) { tot += Math.abs(v); });
    var h = '<div class="t">' + esc(src) + " &rarr; " + esc(dst) + "</div>" +
      '<div class="m">phi = ' + f(e.v, 4) + " at " + (e.L === 1 ? "x" : src) + " = " + f(xin, 3) + "<br>" +
      Math.round(100 * Math.abs(e.v) / (tot || 1)) + "% of the |inputs| into " + esc(dst.split(" ")[0]) + "</div>";
    var ce = cat && cat[layer] && cat[layer].edges[e.j] && cat[layer].edges[e.j][e.i];
    if (ce) {
      var gx = e.L === 1 ? cat.enc.grid : cat.head.grid[e.i];
      h += '<div class="m" style="margin-top:6px">' + esc(ce.exact) + "</div>" +
        '<div class="m">cubic in ' + esc(src) + ": " + esc(ce.cubic_str) + " (R2 " + ce.cubic_r2.toFixed(3) + ")</div>" + sparkline(gx, ce.curve, xin, 300, 70);
    }
    return h;
  }
  function inputTip(A, i) {
    var at = A.attribution, wk = at.week || {};
    var h = '<div class="t">' + esc(A.features[i]) + "</div><div class='m'>raw " + (A.raw[i] === null ? "missing (enters as 0)" : f(A.raw[i], 4)) +
      "<br>standardised " + f(A.zraw[i], 2) + (fin(A.zraw[i]) && Math.abs(A.zraw[i]) > 6 ? " (clipped to " + f(A.z[i], 1) + ")" : "") +
      "<br>sum |phi| into the latent " + f(at.l1_abs[i], 3, false) +
      "<br>sigma if at training mean: " + f(-at.mean_ablation_sigma_pct[i], 3) + " pp" +
      "<br>median if at training mean: " + f(-at.mean_ablation_q50_pct[i], 3) + " pp";
    if (at.week_ablation_dln_sigma) h += "<br>week mean " + f(wk.mean_raw[i], 3) + "; moving it there changes sigma " + f(100 * (Math.exp(-at.week_ablation_dln_sigma[i]) - 1), 1) + "%";
    return h + "</div>";
  }
  function latentTip(A, seed, j) {
    var lat = A.latent, h = '<div class="t">z' + (j + 1) + " (seed " + (S.seed + 1) + ")</div><div class='m'>value " + f(seed.latent[j], 4) +
      "<br>bias " + f(seed.latent_bias[j], 4) + "<br>sd units " + f((seed.latent[j] - lat.zmu[j]) / lat.zsd[j], 2) + " (mean " + f(lat.zmu[j], 3) + ", sd " + f(lat.zsd[j], 3) + ")";
    if (lat.zhat_next) h += "<br>dynamics (seed 1) next: " + f(lat.zhat_next[j], 4) + " = " + f((lat.zhat_next[j] - lat.zmu[j]) / lat.zsd[j], 2) + " sd";
    return h + "</div>";
  }
  function outTip(A, seed, k, qn) {
    return '<div class="t">' + qn[k] + " head</div><div class='m'>raw o" + k + " (seed " + (S.seed + 1) + ") " + f(seed.head_raw[k], 4) +
      "<br>quantile (seed, model units) " + f(seed.quantiles_model[k], 4) + "<br>2-seed mean x scale: " + f(A.quantiles_pct[k], 4) + "%" +
      "<br>stored in the line: " + f(A.stored_quantiles_pct[k], 4) + "%" +
      "<br>" + (k === 3 ? "q50 = o3" : k > 3 ? qn[k] + " = " + qn[k - 1] + " + softplus(o" + k + ")" : qn[k] + " = " + qn[k + 1] + " - softplus(o" + k + ")") + "</div>";
  }

  // ------------------------------------------------------------------ thinking, Claude, H-LIVE
  function renderThinking() {
    var a = S.act; if (!a) return;
    $("think-id").textContent = a.id;
    $("think").innerHTML = (a.thinking || []).map(function (t) { return "<li><b>" + esc(THINK_LABEL[t.k] || t.k) + "</b>" + esc(t.text) + "</li>"; }).join("");
  }
  function renderClaude() {
    var c = S.state.claude || {}, v = c.latest_view;
    if (!v) { $("claude-cap").textContent = "Claude's read"; $("claude-view").textContent = "No Claude forecast logged yet."; $("claude-band").textContent = ""; return; }
    $("claude-cap").textContent = "Claude's read, " + hhmm(v.logged_at) + " UTC";
    $("claude-view").textContent = v.view;
    var L = line(), parts = [];
    ["latest_1h", "latest_24h"].forEach(function (k) {
      var x = c[k]; if (!x) return;
      parts.push(x.horizon + " " + hhmm(x.window_start) + ": q10 " + f(x.q10) + " / q50 " + f(x.q50) + " / q90 " + f(x.q90) + ", sigma " + f(x.sigma, 2, false));
    });
    if (L) parts.push("net sigma 1h " + f(L.net["1h"].sigma_pct, 2, false) + ", 24h " + f(L.net["24h"].sigma_pct, 2, false));
    var cur = c.current_1h;
    if (cur && (!c.latest_1h || cur.id !== c.latest_1h.id)) parts.push("this hour (logged " + hhmm(cur.logged_at) + "): 80% " + f(cur.q10) + " / " + f(cur.q90));
    $("claude-band").textContent = parts.join(" | ");
  }
  function renderHlive() {
    var h = S.state.hlive || {}, t = $("hlive-tbl");
    if (h.error || !h.live1) { t.innerHTML = ""; $("hlive-note").textContent = h.error ? "score error: " + h.error : "No scored pairs yet."; return; }
    var r = "<tr><th></th><th>n</th><th>Claude</th><th>Net</th><th>d</th><th>t</th><th>Verdict</th></tr>";
    ["1h", "24h"].forEach(function (k) {
      var x = h.live1[k];
      r += "<tr><td>H-LIVE1 " + k + "</td><td>" + x.n + "/" + h.min_live1 + '<div class="prog"><i style="width:' + Math.min(100, 100 * x.n / h.min_live1) + '%"></i></div></td><td>' +
        f(x.loss_claude, 3, false) + "</td><td>" + f(x.loss_net, 3, false) + "</td><td>" + f(x.mean_d, 3) + "</td><td>" + f(x.t, 2) + '</td><td><span class="verdict">' + esc(x.verdict) + "</span></td></tr>";
    });
    var l2 = h.live2 || {}, ds = l2.descriptive || {};
    r += "<tr><td>H-LIVE2</td><td>" + (l2.n || 0) + "/" + h.min_live2 + '<div class="prog"><i style="width:' + Math.min(100, 100 * (l2.n || 0) / h.min_live2) + '%"></i></div></td><td colspan="2">log-score diff</td><td>' +
      f(l2.mean_d, 3) + "</td><td>" + f(l2.t, 2) + '</td><td><span class="verdict">' + esc(l2.verdict || "pending") + "</span></td></tr>";
    t.innerHTML = r;
    $("hlive-note").textContent = "Pinball loss over 10/50/90 in % log return; d = Claude - net (negative favours Claude); t = Newey-West over daily means. " +
      (ds.rate !== undefined ? "H-LIVE2 descriptive sign-hit " + f(ds.rate, 2, false) + " vs direction-blind " + f(ds.null_rate_direction_blind, 2, false) + ". " : "") +
      "Verdicts count only at the registered minimum sample. " + h.pairs + " pairs from " + h.claude_forecasts + " forecasts.";
  }

  // ------------------------------------------------------------------ regime strip
  function renderRegime() {
    var R = S.state.regime, el = $("regime");
    if (!R) { el.className = "strip empty"; el.textContent = "Regime and players: not in research/live/LATEST.md yet (agent V2 is adding the section)."; return; }
    if (R.error || !R.rows.length) { el.className = "strip empty"; el.textContent = R.title + (R.error ? "" : " (no rows)"); return; }
    el.className = "strip";
    var b = R.rows.find(function (r) { return r.symbol === "BTC"; }) || R.rows[0];
    function m(v) { return fin(v) ? (v >= 1e5 ? (v / 1e6).toFixed(1) + "M" : (v / 1e3).toFixed(0) + "k") : "-"; }
    function bar(l, s) {
      var t = (l || 0) + (s || 0); if (!t) return "";
      return '<div class="liqbar" title="longs liquidated (left) vs shorts (right)"><i class="l" style="width:' + (100 * l / t).toFixed(0) + '%"></i><i class="s" style="width:' + (100 * s / t).toFixed(0) + '%"></i></div>';
    }
    var pat = R.pattern || {};
    var others = R.rows.filter(function (r) { return r !== b; }).map(function (r) { return r.symbol + " " + r.cell; }).join(", ");
    var cells = [
      ["v0 cell", esc(b.cell) + " <small>" + esc(b.cell_label || "") + "</small>"],
      ["24h price", f(b.ret_24h_pct, 1) + "%"],
      ["Funding", f(b.funding_rate_pct, 4) + "% <small>p" + (b.funding_pct_90d === null ? "-" : b.funding_pct_90d) + " of 90d</small>"],
      ["OI 24h / 72h", f(b.oi_chg_24h_pct, 1) + "% / " + f(b.oi_chg_72h_pct, 1) + "%"],
      ["Liq 4h long / short", m(b.liq_4h_long_usd) + " / " + m(b.liq_4h_short_usd) + bar(b.liq_4h_long_usd, b.liq_4h_short_usd)],
      ["Liq 24h long / short", m(b.liq_24h_long_usd) + " / " + m(b.liq_24h_short_usd) + (b.liq_24h_partial ? " <small>partial</small>" : "") + bar(b.liq_24h_long_usd, b.liq_24h_short_usd)],
      ["DVOL", (fin(b.dvol) ? b.dvol.toFixed(1) : "-") + " <small>p" + (b.dvol_pct_90d === null ? "-" : b.dvol_pct_90d) + " of 90d</small>"],
      ["Pattern" + (R.as_of ? " (" + R.as_of.slice(11, 16) + ")" : "") + (R.last_seen ? ", last seen " + hhmm(R.last_seen) : ""), '<span class="pat">' + esc(pat[b.symbol] || "-") + "</span>" + (others ? " <small>" + esc(others) + "</small>" : "")]
    ];
    el.innerHTML = cells.map(function (c) { return '<div class="cellx"><span class="k">' + c[0] + '</span><span class="v">' + c[1] + "</span></div>"; }).join("");
  }

  // ------------------------------------------------------------------ latent and realised charts
  function baseLayout(C) {
    return { paper_bgcolor: C.surf, plot_bgcolor: C.surf, font: { family: "IBM Plex Mono, ui-monospace, monospace", size: 11, color: C.ink2 },
      margin: { l: 40, r: 12, t: 8, b: 28 }, hovermode: "x unified",
      xaxis: { type: "date", gridcolor: C.grid, linecolor: C.grid, tickformat: "%H:%M\n%b %d", hoverformat: "%b %d %H:%M UTC" },
      yaxis: { gridcolor: C.grid, zerolinecolor: C.ink3, zerolinewidth: 1 },
      legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } } };
  }
  function colors() {
    return { surf: css("--surface"), ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"), grid: css("--grid"), v1: css("--v1"), neg: css("--neg"),
      lat: [css("--lat1"), css("--lat2"), css("--lat3"), css("--lat4")] };
  }
  function drawLatent() {
    if (!window.Plotly || !S.state || !S.state.history.length) return;
    var C = colors(), H = S.state.history, mu = S.state.latent.zmu, sd = S.state.latent.zsd, L = line();
    if (!mu || !sd) return;
    var tr = [];
    for (var j = 0; j < 4; j++) {
      var xs = [], ys = [];
      H.forEach(function (h) { if (h.z) { xs.push(Date.parse(h.forecast_hour)); ys.push((h.z[j] - mu[j]) / sd[j]); } });
      tr.push({ type: "scatter", mode: "lines", x: xs, y: ys, name: "z" + (j + 1), line: { color: C.lat[j], width: 2 }, hovertemplate: "%{y:+.2f} sd" });
      if (L && L.latent && L.latent.zhat_next) {
        var t0 = Date.parse(L.forecast_hour);
        tr.push({ type: "scatter", mode: "lines+markers", x: [t0, t0 + HOUR], y: [(L.latent.z[j] - mu[j]) / sd[j], (L.latent.zhat_next[j] - mu[j]) / sd[j]],
          line: { color: C.lat[j], width: 2, dash: "dot" }, marker: { size: [0, 8], color: C.lat[j], line: { color: C.surf, width: 2 } },
          showlegend: false, name: "z" + (j + 1) + " next", hovertemplate: "next %{y:+.2f} sd" });
      }
    }
    var lay = baseLayout(C);
    lay.yaxis.title = { text: "sd", font: { size: 10 } };
    Plotly.react($("latent"), tr, lay, { displaylogo: false, responsive: true, displayModeBar: false });
  }
  function drawRealised() {
    if (!window.Plotly || !S.state) return;
    var C = colors(), R = S.state.realised || [];
    if (!R.length) { $("real-note").textContent = "No scored hours yet."; return; }
    var x = R.map(function (r) { return Date.parse(r.forecast_hour); });
    var b95lo = R.map(function (r) { return r.aci_1h["95"][0] / r.sigma_1h; }), b95hi = R.map(function (r) { return r.aci_1h["95"][1] / r.sigma_1h; });
    var b80lo = R.map(function (r) { return r.aci_1h["80"][0] / r.sigma_1h; }), b80hi = R.map(function (r) { return r.aci_1h["80"][1] / r.sigma_1h; });
    var W = 0.62 * HOUR;
    var tr = [
      { type: "bar", x: x, base: b95lo, y: b95hi.map(function (v, i) { return v - b95lo[i]; }), width: W, marker: { color: rgba(C.v1, 0.16) }, name: "ACI 95%", hoverinfo: "skip" },
      { type: "bar", x: x, base: b80lo, y: b80hi.map(function (v, i) { return v - b80lo[i]; }), width: W * 0.55, marker: { color: rgba(C.v1, 0.4) }, name: "ACI 80%", hoverinfo: "skip" },
      { type: "scatter", mode: "markers", x: x, y: R.map(function (r) { return r.z_1h; }), name: "realised",
        marker: { size: 8, symbol: R.map(function (r) { return r.hit80_1h ? "circle" : "x"; }), color: R.map(function (r) { return r.hit80_1h ? C.ink : C.neg; }),
          line: { color: C.surf, width: 1 } },
        customdata: R.map(function (r) { return [r.r_1h, r.hit80_1h ? "in" : "out", r.hit95_1h ? "in" : "out", r.id]; }),
        hovertemplate: "z %{y:+.2f}, r %{customdata[0]:+.3f}%<br>ACI80 %{customdata[1]}, ACI95 %{customdata[2]}<extra>%{customdata[3]}</extra>" }
    ];
    var lay = baseLayout(C);
    lay.barmode = "overlay"; lay.hovermode = "closest";
    lay.yaxis.title = { text: "r / sigma", font: { size: 10 } };
    Plotly.react($("realised"), tr, lay, { displaylogo: false, responsive: true, displayModeBar: false });
    var n = R.length, h80 = R.filter(function (r) { return r.hit80_1h; }).length, h95 = R.filter(function (r) { return r.hit95_1h; }).length;
    $("real-note").textContent = "Inside ACI 80%: " + h80 + "/" + n + " (" + Math.round(100 * h80 / n) + "%), inside 95%: " + h95 + "/" + n +
      " (" + Math.round(100 * h95 / n) + "%). Circles are inside the 80% band, crosses outside.";
  }

  // ------------------------------------------------------------------ footer
  function renderFooter() {
    var L = line(); if (!L) return;
    var m = L.models, v2 = v2line();
    $("foot-models").textContent = "pnn_1h " + m.pnn_1h.refit_month + " pt " + m.pnn_1h.pt_sha256 + " | pnn_1d " + m.pnn_1d.refit_month + " pt " + m.pnn_1d.pt_sha256 +
      " | field " + (m.field && m.field.file) + " | code " + m.code_sha256 + " | " + m.tool_version + (v2 ? " | v2 cfg " + v2.config_hash : " | v2 not present") +
      " | lines " + S.state.sources.v1_lines;
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    var t = store("btcmon.theme"), qt = new URLSearchParams(location.search).get("theme");
    if (qt && THEMES.indexOf(qt) >= 0) t = qt;
    if (t && THEMES.indexOf(t) >= 0) applyTheme(t);
    loadState();
    loadKlines();
    connect();
    setInterval(loadState, POLL_MS);
    document.addEventListener("visibilitychange", function () { if (!document.hidden) loadState(); });
    window.addEventListener("resize", function () { drawNet(false); scheduleChart(true); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
