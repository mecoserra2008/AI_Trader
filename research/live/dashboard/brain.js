/* Brain map of the frozen 1h BTC network, firing on intra-hour ticks (an intra-hour nowcast, not the official forecast).
   Data: /api/brain/net (both seeds' weights), /api/brain/base (frozen inputs of the newest v1 line + recipe, polled),
   /api/kline (1m candles to seed the partial hour), Bybit public websocket publicTrade.BTCUSDT + tickers.BTCUSDT.
   Forward pass and recipe: brain_core.js (tested equal to src/ai_trader/livedash/brain.py). No writes, no POST.
   CPU: passes are throttled (COMPUTE_MS), the static map is cached on an offscreen canvas, animation runs only while
   something is moving (<= 20 fps), raster / meters / panels redraw at 1 Hz, nothing runs while the tab is hidden. */
(function () {
  "use strict";
  var C = window.BrainCore;
  var WS_URL = "wss://stream.bybit.com/v5/public/linear";
  var COMPUTE_MS = 160, IDLE_MS = 1000, FRAME_MS = 50, WIN_S = 900, BASE_POLL_MS = 15000, PING_MS = 20000;
  var IN_THR = 0.01, UNIT_THR = 0.01, EMIT_MIN = 0.02, GLOW_TAU = 0.9, HEAT_TAU = 1.2, RATE_TAU = 60;
  var FAM_ORDER = ["volatility", "returns", "jumps", "funding", "calendar"];
  var FAM_VAR = { volatility: "--v1", returns: "--v2", jumps: "--crit", funding: "--lat3", calendar: "--claude" };
  var SEED_NAME = ["A", "B"];

  function $(id) { return document.getElementById(id); }
  function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888"; }
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } return null; }
  function fmt(v, d) { return (v === null || v === undefined || !isFinite(v)) ? "n/a" : (+v).toFixed(d === undefined ? 2 : d); }
  function sgn(v, d) { return (v >= 0 ? "+" : "") + fmt(v, d); }
  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function getJSON(u) { return fetch(u, { cache: "no-store" }).then(function (r) { if (!r.ok) throw new Error(u + " " + r.status); return r.json(); }); }
  var reduced = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };

  // ------------------------------------------------------------------ state
  var NET = null, BASE = null, BASES = {}, ST = null, US = null;
  var FW = null, FW0 = null, FWT = 0, X = null, INFO = null;
  var clockOff = 0, lastTickTs = 0, dirty = true, lastCompute = 0, passes = 0, passT0 = performance.now();
  var paused = false, replay = null, priceSec = [], frames = [];
  var nodes = [], edges = [], rows = [], byKey = {}, glow, heat, rate, spikes = [], raster, rasterT;
  var W = 0, H = 0, DPR = 1, staticCv = null, colors = {}, hoverItem = null, lastWhy = 0, cpuBusy = 0;

  // ------------------------------------------------------------------ theme
  var THEMES = ["auto", "light", "dark"];
  function applyTheme(t) {
    if (t === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", t);
    $("theme").textContent = "Theme: " + t;
    readColors(); layout(); if (NET) legend(); kick();
  }
  $("theme").addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme") || "auto";
    var nxt = THEMES[(THEMES.indexOf(cur) + 1) % 3];
    store("btcmon.theme", nxt); applyTheme(nxt);
  });
  function readColors() {
    ["--bg", "--surface", "--surface-2", "--rule", "--grid", "--ink", "--ink-2", "--ink-3", "--v1", "--v2", "--pos", "--neg", "--up", "--down",
     "--crit", "--lat1", "--lat2", "--lat3", "--lat4", "--claude", "--field", "--warn"].forEach(function (k) { colors[k] = css(k); });
    colors.dark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (!document.documentElement.getAttribute("data-theme") && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function rgba(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    if (!isFinite(n)) return hex;
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + clamp(a, 0, 1).toFixed(3) + ")";
  }

  // ------------------------------------------------------------------ anatomy
  function build() {
    nodes = []; edges = []; rows = []; byKey = {};
    var F = NET.features, nS = NET.seeds.length, k = NET.latent;
    function add(n) { n.i = nodes.length; nodes.push(n); byKey[n.key] = n; return n; }
    FAM_ORDER.forEach(function (fam) {
      F.forEach(function (c, i) { if (NET.family[c] === fam) add({ key: "in" + i, kind: "in", idx: i, label: c, family: fam }); });
    });
    for (var s = 0; s < nS; s++) for (var j = 0; j < k; j++) add({ key: "z" + s + "_" + j, kind: "z", seed: s, idx: j, label: SEED_NAME[s] + ".z" + (j + 1) });
    for (s = 0; s < nS; s++) for (j = 0; j < 7; j++) add({ key: "o" + s + "_" + j, kind: "o", seed: s, idx: j, label: SEED_NAME[s] + ".o" + (j + 1) });
    for (j = 0; j < 7; j++) add({ key: "q" + j, kind: "q", idx: j, label: C.QNAMES[j] });
    for (s = 0; s < nS; s++) {
      for (j = 0; j < k; j++) for (var i = 0; i < F.length; i++)
        edges.push({ kind: "enc", seed: s, o: j, i: i, a: byKey["in" + i], b: byKey["z" + s + "_" + j] });
      for (var o = 0; o < 7; o++) for (j = 0; j < k; j++)
        edges.push({ kind: "head", seed: s, o: o, i: j, a: byKey["z" + s + "_" + j], b: byKey["o" + s + "_" + o] });
      for (var q = 0; q < 7; q++) {  // q_k depends on o_3 and the softplus gaps between o_3 and o_k
        var deps = [3];
        if (q > 3) for (var d = 4; d <= q; d++) deps.push(d);
        if (q < 3) for (d = q; d <= 2; d++) deps.push(d);
        deps.forEach(function (dd) { edges.push({ kind: "out", seed: s, o: q, i: dd, a: byKey["o" + s + "_" + dd], b: byKey["q" + q] }); });
      }
    }
    edges.forEach(function (e, n) { e.n = n; });
    // raster rows: changing inputs first, then frozen inputs, latent, head, quantiles
    var ch = BASE ? BASE.changing : [];
    nodes.filter(function (n) { return n.kind === "in" && ch.indexOf(n.label) >= 0; }).forEach(function (n) { rows.push(n); });
    nodes.filter(function (n) { return n.kind === "in" && ch.indexOf(n.label) < 0; }).forEach(function (n) { rows.push(n); });
    nodes.filter(function (n) { return n.kind !== "in"; }).forEach(function (n) { rows.push(n); });
    rows.forEach(function (n, r) { n.row = r; });
    glow = new Float32Array(nodes.length); rate = new Float32Array(nodes.length); heat = new Float32Array(edges.length);
    raster = new Float32Array(nodes.length * WIN_S); rasterT = new Float64Array(WIN_S).fill(-1);
  }

  function layout() {
    var cv = $("brain"), r = cv.getBoundingClientRect();
    if (!NET || r.width < 10) return;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
    var F = NET.features.length, k = NET.latent;
    // sensory arc (left): inputs grouped by family, gaps between families
    var cx = W * 0.62, cy = H * 0.5, R = Math.min(W * 0.46, H * 0.62), span = 2.15, gaps = FAM_ORDER.length - 1;
    var ins = nodes.filter(function (n) { return n.kind === "in"; });
    var step = span / (F - 1 + gaps * 0.9), a = Math.PI - span / 2, prevFam = null;
    ins.forEach(function (n) {
      if (prevFam && n.family !== prevFam) a += step * 0.9;
      n.x = cx + R * Math.cos(a); n.y = cy - R * Math.sin(a) * 0.74; n.ang = a; n.r = 6; prevFam = n.family; a += step;
    });
    // latent core: two hemispheres (seeds) around the centre
    var zx = W * 0.47;
    nodes.forEach(function (n) {
      if (n.kind === "z") {
        var top = n.seed === 0, t = (n.idx + 0.5) / k;
        n.x = zx + 18 * Math.sin(Math.PI * t); n.y = top ? H * (0.10 + 0.30 * t) : H * (0.60 + 0.30 * t); n.r = 11;
      } else if (n.kind === "o") {
        var t2 = (n.idx + 0.5) / 7, top2 = n.seed === 0;
        n.x = W * 0.70; n.y = top2 ? H * (0.07 + 0.37 * t2) : H * (0.56 + 0.37 * t2); n.r = 6;
      } else if (n.kind === "q") {
        n.x = W * 0.90 - 10 * Math.sin(Math.PI * (n.idx + 0.5) / 7); n.y = H * (0.86 - 0.72 * n.idx / 6); n.r = 9;
      }
    });
    edges.forEach(function (e) {
      var ax = e.a.x, ay = e.a.y, bx = e.b.x, by = e.b.y;
      var mx = (ax + bx) / 2, my = (ay + by) / 2, bend = e.kind === "enc" ? 0.10 : 0.05;
      e.cx = mx + (by - ay) * bend * (e.seed ? -1 : 1); e.cy = my - (bx - ax) * bend * (e.seed ? -1 : 1);
      e.mx = 0.25 * ax + 0.5 * e.cx + 0.25 * bx; e.my = 0.25 * ay + 0.5 * e.cy + 0.25 * by;  // bezier midpoint = bead
    });
    drawStatic();
  }

  function famColor(f) { return colors[FAM_VAR[f]] || colors["--ink-3"]; }
  function nodeColor(n) {
    if (n.kind === "in") return famColor(n.family);
    if (n.kind === "z") return colors["--lat" + (n.idx + 1)];
    if (n.kind === "o") return n.seed ? colors["--v2"] : colors["--v1"];
    return colors["--ink-2"];
  }

  function drawStatic() {
    staticCv = staticCv || document.createElement("canvas");
    staticCv.width = Math.round(W * DPR); staticCv.height = Math.round(H * DPR);
    var g = staticCv.getContext("2d");
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
    g.clearRect(0, 0, W, H);
    g.lineWidth = 0.6;
    edges.forEach(function (e) {
      g.strokeStyle = rgba(colors["--ink-3"], e.kind === "enc" ? 0.10 : 0.16);
      g.beginPath(); g.moveTo(e.a.x, e.a.y); g.quadraticCurveTo(e.cx, e.cy, e.b.x, e.b.y); g.stroke();
      if (e.kind !== "out") { g.fillStyle = rgba(colors["--ink-3"], 0.35); g.beginPath(); g.arc(e.mx, e.my, 1.2, 0, 6.2832); g.fill(); }
    });
    g.font = "500 11px " + css("--mono");
    g.textBaseline = "middle";
    nodes.forEach(function (n) {
      g.fillStyle = colors["--ink-2"];
      if (n.kind === "in") { g.textAlign = "right"; g.fillText(n.label, n.x - 10, n.y); }
      else if (n.kind === "q") { g.textAlign = "left"; g.fillText(n.label, n.x + 14, n.y); }
      else if (n.kind === "z") { g.textAlign = "right"; g.fillText(n.label, n.x - 15, n.y); }
      else if (n.kind === "o") { g.textAlign = "left"; g.fillStyle = colors["--ink-3"]; g.font = "400 9px " + css("--mono"); g.fillText("o" + (n.idx + 1), n.x + 9, n.y); g.font = "500 11px " + css("--mono"); }
    });
    g.font = "600 10px " + css("--sans");
    g.fillStyle = colors["--ink-3"]; g.textAlign = "center";
    var ins = nodes.filter(function (n) { return n.kind === "in"; });
    var lx = Math.min.apply(null, ins.map(function (n) { return n.x; }));
    g.textAlign = "left"; g.fillText("SENSORY INPUTS (19)", 8, H - 8); g.textAlign = "center";
    g.fillText("LATENT CORE", nodes.filter(function (n) { return n.kind === "z"; })[0].x, H * 0.5);
    g.fillText("hemisphere A (seed 0)", W * 0.47, H * 0.045);
    g.fillText("hemisphere B (seed 1)", W * 0.47, H * 0.975);
    g.fillText("HEAD o1-o7", W * 0.70, H * 0.5);
    g.fillText("QUANTILES", W * 0.90, H * 0.05);
    // family labels along the arc
    FAM_ORDER.forEach(function (fam) {
      var fs = ins.filter(function (n) { return n.family === fam; });
      if (!fs.length) return;
      var mid = fs[Math.floor(fs.length / 2)];
      g.save(); g.fillStyle = famColor(fam); g.textAlign = "left"; g.font = "600 10px " + css("--sans");
      var fy = (fs[0].y + fs[fs.length - 1].y) / 2;
      g.fillText(fam.toUpperCase(), Math.max(4, lx - 150), fy); g.restore();
    });
  }

  // ------------------------------------------------------------------ the forward pass on the live state
  function nowMs() { return Date.now() + clockOff; }
  function unitScale(n) {
    if (!US) return 1;
    if (n.kind === "z") return US.z_sd[n.seed][n.idx];
    if (n.kind === "o") return US.o_sd[n.seed][n.idx];
    if (n.kind === "q") return US.q_sd[n.idx];
    return 1;
  }
  function unitMean(n) {
    if (!US) return 0;
    if (n.kind === "z") return US.z_mu[n.seed][n.idx];
    if (n.kind === "o") return US.o_mu[n.seed][n.idx];
    if (n.kind === "q") return US.q_mu[n.idx];
    return US.x_mu[n.idx];
  }
  function act(fw, n) {
    if (n.kind === "in") return fw.x[n.idx];
    if (n.kind === "z") return fw.seeds[n.seed].z[n.idx];
    if (n.kind === "o") return fw.seeds[n.seed].o[n.idx];
    return fw.q[n.idx];
  }
  function edgeOut(fw, e) {
    if (e.kind === "enc") return fw.seeds[e.seed].enc[e.o][e.i];
    if (e.kind === "head") return fw.seeds[e.seed].head[e.o][e.i];
    return fw.seeds[e.seed].q[e.o];
  }
  function edgeSd(e) {
    if (!US) return 1;
    if (e.kind === "enc") return US.enc_edge_sd[e.seed][e.o][e.i];
    if (e.kind === "head") return US.head_edge_sd[e.seed][e.o][e.i];
    return US.q_sd[e.o];
  }

  /* mode 'live' (records the raster and rates) or 'replay' (visual only); each keeps its own previous pass. */
  var PREV = { live: null, replay: null };
  function compute(t, state, base, mode, visual) {
    var r = C.nowcastInputs(base, state, t);
    var fw = C.forward(NET, r[0]);
    passes++;
    var pv = PREV[mode];
    if (pv && pv.base === base.line_id) fire(t, fw, pv.fw, mode, visual);
    PREV[mode] = { fw: fw, base: base.line_id };
    if ((mode === "replay") === !!replay) { FW = fw; X = r[0]; INFO = r[1]; FWT = t; if (base !== BASE) FW0 = C.forward(NET, base.inputs.map(function (v) { return v === null ? NaN : v; })); }
    return fw;
  }

  function fire(t, fw, old, mode, visual) {
    var sec = Math.floor(t / 1000), col = sec % WIN_S, fired = {}, rec = mode === "live";
    if (rec && rasterT[col] !== sec) { for (var r0 = 0; r0 < nodes.length; r0++) raster[r0 * WIN_S + col] = 0; rasterT[col] = sec; }
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i], d = (act(fw, n) - act(old, n)) / unitScale(n), ad = Math.abs(d);
      if (ad > (n.kind === "in" ? IN_THR : UNIT_THR)) {
        fired[n.key] = d;
        var amp = clamp(ad / 0.5, 0.08, 1);
        if (rec) { raster[n.i * WIN_S + col] = Math.max(raster[n.i * WIN_S + col], amp); rate[i] += 1; }
        if (visual && (reduced.matches || n.kind === "in")) glow[i] = Math.max(glow[i], amp);
      }
    }
    if (!visual) return;
    for (var k = 0; k < edges.length; k++) {
      var e = edges[k];
      if (fired[e.a.key] === undefined) continue;
      var dphi = (edgeOut(fw, e) - edgeOut(old, e)) / edgeSd(e), a = clamp(Math.abs(dphi) / 0.25, 0, 1);
      if (a < EMIT_MIN) continue;
      heat[k] = Math.max(heat[k], a);
      var tgtAmp = clamp(Math.abs(fired[e.b.key] || 0) / 0.5, 0, 1);
      if (reduced.matches) continue;
      var dur = clamp(1.4 / (0.6 + 3 * a), 0.25, 1.6) * (e.kind === "enc" ? 1 : 0.6);
      spikes.push({ e: e, t0: performance.now() / 1000 + (e.kind === "enc" ? 0 : e.kind === "head" ? 0.35 : 0.6), dur: dur, amp: a,
                    pos: dphi >= 0, tgt: e.b.i, tgtAmp: Math.max(tgtAmp, 0.15 * a) });
    }
    if (spikes.length > 900) spikes.splice(0, spikes.length - 900);
    kick();
  }

  // ------------------------------------------------------------------ animation
  var raf = 0, lastFrame = 0, lastDecay = performance.now() / 1000;
  function kick() { if (!raf && !document.hidden) raf = requestAnimationFrame(frame); }
  function frame(ts) {
    raf = 0;
    if (ts - lastFrame < FRAME_MS) { raf = requestAnimationFrame(frame); return; }
    lastFrame = ts;
    var a0 = performance.now();
    try { frameBody(ts); } finally { cpuBusy += performance.now() - a0; }
  }
  function frameBody(ts) {
    var now = ts / 1000, dt = Math.min(now - lastDecay, 0.5); lastDecay = now;
    var gk = Math.exp(-dt / GLOW_TAU), hk = Math.exp(-dt / HEAT_TAU), active = false, i;
    if (!paused) {
      for (i = 0; i < glow.length; i++) { glow[i] *= gk; if (glow[i] > 0.04) active = true; }
      for (i = 0; i < heat.length; i++) { heat[i] *= hk; if (heat[i] > 0.05) active = true; }
    }
    drawBrain(now);
    if (!paused) {
      var keep = [];
      for (i = 0; i < spikes.length; i++) {
        var s = spikes[i];
        if (now - s.t0 >= s.dur) { glow[s.tgt] = Math.max(glow[s.tgt], s.tgtAmp); continue; }
        keep.push(s);
      }
      spikes = keep;
    }
    if (replay) { stepReplay(dt); active = true; }
    if (!paused && (active || spikes.length)) raf = requestAnimationFrame(frame);
  }

  function drawBrain(now) {
    var cv = $("brain"), g = cv.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, cv.width, cv.height);
    if (staticCv) g.drawImage(staticCv, 0, 0);
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
    var pos = colors["--pos"], neg = colors["--neg"], i;
    // hot edges and their beads (the hidden polynomial units)
    for (i = 0; i < edges.length; i++) {
      var h = heat[i], e = edges[i];
      if (h < 0.02 && hoverItem !== e) continue;
      var v = FW ? edgeOut(FW, e) : 0, col = v >= 0 ? pos : neg;
      g.strokeStyle = rgba(col, hoverItem === e ? 0.95 : 0.15 + 0.6 * h);
      g.lineWidth = hoverItem === e ? 2 : 0.6 + 1.6 * h;
      g.beginPath(); g.moveTo(e.a.x, e.a.y); g.quadraticCurveTo(e.cx, e.cy, e.b.x, e.b.y); g.stroke();
      if (e.kind !== "out") { g.fillStyle = rgba(col, 0.35 + 0.6 * h); g.beginPath(); g.arc(e.mx, e.my, 1.4 + 2.6 * h, 0, 6.2832); g.fill(); }
    }
    // spikes
    if (!reduced.matches) {
      for (i = 0; i < spikes.length; i++) {
        var s = spikes[i], p = (now - s.t0) / s.dur;
        if (p < 0 || p > 1) continue;
        var E = s.e, q1 = 1 - p;
        var x = q1 * q1 * E.a.x + 2 * q1 * p * E.cx + p * p * E.b.x, y = q1 * q1 * E.a.y + 2 * q1 * p * E.cy + p * p * E.b.y;
        var c2 = s.pos ? pos : neg;
        g.fillStyle = rgba(c2, 0.25 * s.amp + 0.1); g.beginPath(); g.arc(x, y, 3 + 5 * s.amp, 0, 6.2832); g.fill();
        g.fillStyle = rgba(c2, 0.55 + 0.45 * s.amp); g.beginPath(); g.arc(x, y, 1.2 + 1.8 * s.amp, 0, 6.2832); g.fill();
      }
    }
    // nodes: fill = activation vs mean (blue above / red below), halo = recent |change|
    for (i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (n.x === undefined) continue;
      var dev = FW ? (act(FW, n) - unitMean(n)) / (n.kind === "in" ? (US ? US.x_sd[n.idx] : 1) : unitScale(n)) : 0;
      if (glow[i] > 0.01) { g.fillStyle = rgba(nodeColor(n), 0.35 * glow[i]); g.beginPath(); g.arc(n.x, n.y, n.r + 4 + 12 * glow[i], 0, 6.2832); g.fill(); }
      g.fillStyle = colors["--surface"]; g.beginPath(); g.arc(n.x, n.y, n.r, 0, 6.2832); g.fill();
      g.fillStyle = rgba(dev >= 0 ? pos : neg, clamp(Math.abs(dev) / 2.5, 0.05, 0.95)); g.beginPath(); g.arc(n.x, n.y, n.r - 1.5, 0, 6.2832); g.fill();
      var changing = n.kind !== "in" || (BASE && BASE.changing.indexOf(n.label) >= 0);
      g.strokeStyle = nodeColor(n); g.lineWidth = hoverItem === n ? 3 : (changing ? 2 : 1); g.setLineDash(changing ? [] : [2, 2]);
      g.beginPath(); g.arc(n.x, n.y, n.r, 0, 6.2832); g.stroke(); g.setLineDash([]);
    }
  }

  // ------------------------------------------------------------------ raster and meters (1 Hz)
  function sizeCanvas(cv) {
    var r = cv.getBoundingClientRect(), d = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== Math.round(r.width * d) || cv.height !== Math.round(r.height * d)) { cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d); }
    var g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0);
    return { g: g, w: r.width, h: r.height };
  }
  function drawRaster() {
    if (!NET) return;
    var o = sizeCanvas($("raster")), g = o.g, w = o.w, h = o.h, L = 74, top = 4, bot = 16;
    g.clearRect(0, 0, w, h);
    var rh = (h - top - bot) / rows.length, tNow = Math.floor((replay ? replay.t : nowMs()) / 1000), t0 = Math.floor(nowMs() / 1000) - WIN_S + 1;
    g.font = "400 9px " + css("--mono"); g.textBaseline = "middle"; g.textAlign = "right";
    rows.forEach(function (n, r) {
      var y = top + r * rh;
      if (r % 2 === 0) { g.fillStyle = rgba(colors["--grid"], 0.6); g.fillRect(L, y, w - L, rh); }
      if (rh >= 6.5 || r % 2 === 0) { g.fillStyle = nodeColor(n); g.fillText(n.label, L - 4, y + rh / 2); }
    });
    var pw = (w - L) / WIN_S;
    for (var c = 0; c < WIN_S; c++) {
      var sec = rasterT[c];
      if (sec < t0) continue;
      var x = L + (sec - t0) * pw;
      for (var r = 0; r < rows.length; r++) {
        var a = raster[rows[r].i * WIN_S + c];
        if (a <= 0) continue;
        g.fillStyle = rgba(nodeColor(rows[r]), 0.25 + 0.75 * a);
        g.fillRect(x, top + r * rh + 0.5, Math.max(pw, 1.2), Math.max(rh - 1, 1));
      }
    }
    g.fillStyle = colors["--ink-3"]; g.textAlign = "center"; g.textBaseline = "alphabetic";
    for (var m = 0; m <= 15; m += 3) { g.textAlign = m === 0 ? "right" : (m === 15 ? "left" : "center"); g.fillText(m === 0 ? "now" : "-" + m + " min", L + (WIN_S - m * 60) * pw, h - 3); }
    if (replay) { g.strokeStyle = colors["--v2"]; g.lineWidth = 2; var xr = L + (tNow - t0) * pw; g.beginPath(); g.moveTo(xr, top); g.lineTo(xr, h - bot); g.stroke(); }
  }
  function drawMeters() {
    if (!NET) return;
    var o = sizeCanvas($("meters")), g = o.g, w = o.w, h = o.h, top = 4, bot = 16, rh = (h - top - bot) / rows.length;
    g.clearRect(0, 0, w, h);
    var mx = 1;
    rows.forEach(function (n) { mx = Math.max(mx, rate[n.i]); });
    mx = Math.max(10, Math.ceil(mx / 10) * 10);
    rows.forEach(function (n, r) {
      var y = top + r * rh, v = rate[n.i];
      g.fillStyle = rgba(colors["--grid"], 0.8); g.fillRect(0, y + 0.5, w - 34, Math.max(rh - 1, 1));
      g.fillStyle = nodeColor(n); g.fillRect(0, y + 0.5, (w - 34) * v / mx, Math.max(rh - 1, 1));
      if (rh >= 6.5 || r % 2 === 0) { g.fillStyle = colors["--ink-3"]; g.font = "400 9px " + css("--mono"); g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(v.toFixed(0), w - 30, y + rh / 2); }
    });
    g.fillStyle = colors["--ink-3"]; g.font = "400 9px " + css("--mono"); g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillText("spikes / min (0-" + mx + ")", 0, h - 3);
  }

  // ------------------------------------------------------------------ price and quantile panels (1 Hz)
  function drawPrice() {
    if (!BASE || !ST) return;
    var o = sizeCanvas($("price")), g = o.g, w = o.w, h = o.h, L = 56, R = 8, T = 8, B = 18;
    g.clearRect(0, 0, w, h);
    var S = BASE.S_ms, E = S + C.HOUR_MS, t = Math.min(nowMs(), E), P0 = BASE.p0, fq = BASE.frozen_quantiles_pct;
    var lo = P0 * Math.exp(fq[0] / 100), hi = P0 * Math.exp(fq[6] / 100);
    priceSec.forEach(function (p) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); });
    var rem = INFO && FW ? C.remainingPct(FW.q, BASE, INFO.f) : null, P = ST.P;
    if (rem) { lo = Math.min(lo, P * Math.exp(rem[0] / 100)); hi = Math.max(hi, P * Math.exp(rem[6] / 100)); }
    var pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
    function X(tt) { return L + (tt - S) / C.HOUR_MS * (w - L - R); }
    function Y(p) { return T + (hi - p) / (hi - lo) * (h - T - B); }
    g.strokeStyle = colors["--grid"]; g.lineWidth = 1; g.fillStyle = colors["--ink-3"]; g.font = "400 10px " + css("--mono"); g.textAlign = "right"; g.textBaseline = "middle";
    for (var k = 0; k <= 4; k++) { var pv = lo + (hi - lo) * k / 4; g.beginPath(); g.moveTo(L, Y(pv)); g.lineTo(w - R, Y(pv)); g.stroke(); g.fillText(pv.toFixed(0), L - 4, Y(pv)); }
    g.textAlign = "center"; g.textBaseline = "alphabetic";
    for (var m = 0; m <= 60; m += 15) g.fillText(":" + (m < 10 ? "0" : "") + m, X(S + m * 60000), h - 4);
    function cone(t0, p0, qa, qb, frac, col, alpha) {  // band from (t0, p0) widening as sqrt of the elapsed share of [t0, E]
      g.fillStyle = rgba(col, alpha); g.beginPath();
      var n = 24, i, tt, s;
      for (i = 0; i <= n; i++) { tt = t0 + (E - t0) * i / n; s = Math.sqrt(i / n); g.lineTo(X(tt), Y(p0 * Math.exp(qb * s / 100))); }
      for (i = n; i >= 0; i--) { tt = t0 + (E - t0) * i / n; s = Math.sqrt(i / n); g.lineTo(X(tt), Y(p0 * Math.exp(qa * s / 100))); }
      g.closePath(); g.fill();
    }
    cone(S, P0, fq[0], fq[6], 1, colors["--ink-3"], 0.12);
    cone(S, P0, fq[1], fq[5], 1, colors["--ink-3"], 0.16);
    if (rem && !INFO.warm && t < E) { cone(t, P, rem[0], rem[6], 1, colors["--v2"], 0.16); cone(t, P, rem[1], rem[5], 1, colors["--v2"], 0.22); }
    g.strokeStyle = colors["--ink-3"]; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(S), Y(P0)); g.lineTo(X(E), Y(P0)); g.stroke(); g.setLineDash([]);
    g.strokeStyle = colors["--ink"]; g.lineWidth = 1.4; g.beginPath();
    g.moveTo(X(S), Y(P0));
    priceSec.forEach(function (p) { if (p[0] >= S && p[0] <= E) g.lineTo(X(p[0]), Y(p[1])); });
    g.stroke();
    g.fillStyle = colors["--v2"]; g.beginPath(); g.arc(X(t), Y(P), 3.5, 0, 6.2832); g.fill();
    $("price-note").textContent = "Grey: frozen top-of-hour 80% / 90% bands from the hh:00 close " + fmt(P0, 1) +
      ". Orange: the nowcast's 80% / 90% band for the rest of the hour from the live price" + (INFO && INFO.warm ? " (warm-up: first 5 min use the frozen inputs)" : "") + ".";
  }
  function drawQ() {
    if (!BASE || !FW || !INFO) return;
    var o = sizeCanvas($("qplot")), g = o.g, w = o.w, h = o.h, L = 40, R = 52, T = 6, B = 18;
    g.clearRect(0, 0, w, h);
    var nq = C.remainingPct(FW.q, BASE, INFO.f), fz = C.remainingPct(BASE.frozen_quantiles_model, BASE, INFO.f);
    var m = 0.05; nq.concat(fz).forEach(function (v) { m = Math.max(m, Math.abs(v)); }); m *= 1.1;
    function Xp(v) { return L + (v + m) / (2 * m) * (w - L - R); }
    var rh = (h - T - B) / 7;
    g.strokeStyle = colors["--grid"]; g.beginPath(); g.moveTo(Xp(0), T); g.lineTo(Xp(0), h - B); g.stroke();
    g.font = "400 10px " + css("--mono"); g.textBaseline = "middle";
    for (var k = 6; k >= 0; k--) {
      var y = T + (6 - k) * rh + rh / 2;
      g.fillStyle = colors["--ink-3"]; g.textAlign = "right"; g.fillText(C.QNAMES[k], L - 6, y);
      g.strokeStyle = colors["--rule"]; g.beginPath(); g.moveTo(Xp(fz[k]), y); g.lineTo(Xp(nq[k]), y); g.stroke();
      g.strokeStyle = colors["--ink-2"]; g.lineWidth = 1.5; g.beginPath(); g.arc(Xp(fz[k]), y, 4, 0, 6.2832); g.stroke();
      g.fillStyle = colors["--v2"]; g.beginPath(); g.arc(Xp(nq[k]), y, 4, 0, 6.2832); g.fill();
      g.fillStyle = colors["--ink-2"]; g.textAlign = "left"; g.fillText(sgn(nq[k], 2) + "%", w - R + 4, y);
    }
    g.fillStyle = colors["--ink-3"]; g.textAlign = "center"; g.textBaseline = "alphabetic";
    [-m, -m / 2, 0, m / 2, m].forEach(function (v) { g.fillText(sgn(v, 2), Xp(v), h - 4); });
    var sN = (nq[5] - nq[1]) / 2.5631, sF = (fz[5] - fz[1]) / 2.5631;
    var cps = (BASE.checkpoints || []).filter(function (c) { return c.S === BASE.forecast_hour; }).map(function (c) { return "m" + c.minute; });
    $("q-note").textContent = "Filled: nowcast; hollow: frozen x sqrt(1 - f), f = " + fmt(INFO.f, 3) + ". Remaining-hour sigma " + fmt(sN, 3) + "% vs " +
      fmt(sF, 3) + "% (" + sgn(100 * (sN / sF - 1), 0) + "%). H-BR1 checkpoints logged this hour: " + (cps.length ? cps.join(", ") : "none yet") + ".";
  }

  // ------------------------------------------------------------------ what is firing and why (every 2 s)
  function why() {
    if (!BASE || !FW || !FW0 || !INFO) return;
    var out = [], F = NET.features;
    if (INFO.warm) out.push("Warm-up: the first 5 minutes of the hour use the frozen inputs, so the nowcast equals the hh:00 forecast.");
    // the input furthest from its top-of-hour value, and the fastest mover over the last ~10 s
    var best = -1, bd = 0;
    BASE.changing.forEach(function (c) { var i = F.indexOf(c), d = FW.x[i] - FW0.x[i]; if (Math.abs(d) > Math.abs(bd)) { bd = d; best = i; } });
    var NAME = { lu_1: "the partial-hour return input lu_1", lu_m6: "the 6 h mean squared return lu_m6", lu_m24: "the 24 h mean squared return lu_m24",
                 jump24: "the jump flag jump24 (partial-hour range)", since_jump: "hours-since-jump since_jump (partial-hour range)" };
    if (best >= 0 && Math.abs(bd) > 0.005) out.push((NAME[F[best]] || F[best]) + " is <span class=num>" + sgn(bd, 2) + " sd</span> from its top-of-hour value (raw " + fmt(X[best], 3) + ").");
    var hist = null;
    for (var hi = frames.length - 1; hi >= 0; hi--) if (frames[hi].t <= FWT - 10000) { hist = frames[hi]; break; }
    if (hist && hist.x && !replay) {
      var bi = -1, bv = 0;
      BASE.changing.forEach(function (c) { var i = F.indexOf(c), d = FW.x[i] - hist.x[i]; if (Math.abs(d) > Math.abs(bv)) { bv = d; bi = i; } });
      if (bi >= 0 && Math.abs(bv) > 0.02) out.push("In the last " + Math.round((FWT - hist.t) / 1000) + " s " + F[bi] + " moved <span class=num>" + sgn(bv, 2) + " sd</span>" + (INFO.u !== null ? " (partial-hour move " + sgn(100 * INFO.r_p, 3) + "%, hourly-equivalent u = " + fmt(INFO.u, 2) + ")" : "") + ".");
    }
    if (INFO.J) out.push("The partial-hour range crossed 4 s_pre from the open: jump24 and since_jump fired.");
    // most excited latent unit vs the top of the hour, and its push on the 90% quantile
    var bs = 0, bj = 0, bz = 0;
    for (var s = 0; s < NET.seeds.length; s++) for (var j = 0; j < NET.latent; j++) {
      var dz = (FW.seeds[s].z[j] - FW0.seeds[s].z[j]) / (US ? US.z_sd[s][j] : 1);
      if (Math.abs(dz) > Math.abs(bz)) { bz = dz; bs = s; bj = j; }
    }
    if (Math.abs(bz) > 0.005) {
      var z2 = Array.from(FW.seeds[bs].z); z2[bj] = FW0.seeds[bs].z[bj];
      var o2 = C.layer(NET.seeds[bs].head, z2, NET.scale, NET.degree).node, q2 = C.quantilesFromRaw(o2);
      var nS = NET.seeds.length, q90 = FW.q[5] - FW.seeds[bs].q[5] / nS + q2[5] / nS;
      var k = 100 * Math.sqrt(BASE.s2) * Math.sqrt(Math.max(1 - INFO.f, 0)), push = k * (FW.q[5] - q90);
      out.push("Latent " + SEED_NAME[bs] + ".z" + (bj + 1) + " is the most excited unit (<span class=num>" + sgn(bz, 2) + " sd</span> vs the top of the hour); it pushes the 90% quantile " +
        (push >= 0 ? "out" : "in") + " by <span class=num>" + sgn(push, 3) + "%</span> of the remaining-hour return.");
    }
    var nq = C.remainingPct(FW.q, BASE, INFO.f), fz = C.remainingPct(BASE.frozen_quantiles_model, BASE, INFO.f);
    out.push("Net effect: the nowcast's 80% band for the rest of the hour is [" + sgn(nq[1], 2) + ", " + sgn(nq[5], 2) + "]% vs frozen [" + sgn(fz[1], 2) + ", " + sgn(fz[5], 2) + "]%.");
    $("why").innerHTML = out.map(function (s2) { return "<li>" + s2 + "</li>"; }).join("");
  }

  // ------------------------------------------------------------------ hover
  function cheb(c) { return c.map(function (v, d) { return sgn(v, 3) + " T" + (d + 1); }).join(" "); }
  function tipFor(it) {
    if (!FW || !BASE) return "";
    var F = NET.features, s, L;
    if (it.n !== undefined && it.a) {  // edge
      var e = it;
      if (e.kind === "out") return "q " + C.QNAMES[e.o] + " <- " + SEED_NAME[e.seed] + ".o" + (e.i + 1) + "\n" + (NET.head_rule || "") + "\nseed " + SEED_NAME[e.seed] + " " + C.QNAMES[e.o] + " = " + fmt(FW.seeds[e.seed].q[e.o], 4) + " (model units)";
      L = NET.seeds[e.seed][e.kind];
      var xin = e.kind === "enc" ? FW.x[e.i] : FW.seeds[e.seed].z[e.i], inName = e.kind === "enc" ? F[e.i] : SEED_NAME[e.seed] + ".z" + (e.i + 1);
      var outName = e.kind === "enc" ? SEED_NAME[e.seed] + ".z" + (e.o + 1) : SEED_NAME[e.seed] + ".o" + (e.o + 1);
      var v = edgeOut(FW, e), v0 = FW0 ? edgeOut(FW0, e) : v;
      return "polynomial unit  " + inName + " -> " + outName + "\nphi(x) = " + sgn(L.w_res[e.o][e.i], 3) + " silu(x) " + cheb(L.coef[e.o][e.i]) +
        "\n  u = tanh(x/2), T1 = u, T2 = 2u^2-1, T3 = 4u^3-3u, T4 = 8u^4-8u^2+1\nx = " + fmt(xin, 4) + "   phi(x) = " + fmt(v, 4) +
        "\nvs top of hour " + sgn(v - v0, 4) + "   (2025-26 sd " + fmt(edgeSd(e), 4) + ")";
    }
    var n = it;
    if (n.kind === "in") {
      var rec = BASE.recipe.filter(function (r) { return r.feature === n.label; })[0] || {};
      return n.label + "  [" + n.family + "]  " + (rec.changes ? "CHANGES intra-hour" : "frozen until the next close") + "\nraw " + fmt(X[n.idx], 4) +
        "  (top of hour " + fmt(BASE.inputs[n.idx], 4) + ")\nstandardised x = (raw - " + fmt(NET.std_mu[n.idx], 3) + ") / " + fmt(NET.std_sd[n.idx], 3) + " = " + fmt(FW.x[n.idx], 3) + " (clip +-6)\n" + (rec.how || "");
    }
    if (n.kind === "z" || n.kind === "o") {
      s = n.seed; var lay = n.kind === "z" ? "enc" : "head", E = n.kind === "z" ? FW.seeds[s].enc[n.idx] : FW.seeds[s].head[n.idx];
      var names = n.kind === "z" ? F : [0, 1, 2, 3].map(function (j) { return SEED_NAME[s] + ".z" + (j + 1); });
      var top = Array.from(E).map(function (v, i) { return [names[i], v]; }).sort(function (a, b) { return Math.abs(b[1]) - Math.abs(a[1]); }).slice(0, 4);
      return n.label + " = b + sum_i phi_i(x_i) = " + fmt(NET.seeds[s][lay].bias[n.idx], 3) + " + ... = " + fmt(act(FW, n), 4) +
        "\n2025-26 mean " + fmt(unitMean(n), 3) + ", sd " + fmt(unitScale(n), 3) + "; vs top of hour " + sgn((act(FW, n) - (FW0 ? act(FW0, n) : act(FW, n))) / unitScale(n), 2) + " sd" +
        "\nlargest edges: " + top.map(function (t) { return t[0] + " " + sgn(t[1], 3); }).join(", ");
    }
    var rem = C.remainingPct(FW.q, BASE, INFO.f);
    return n.label + " (mean of the 2 seeds) = " + fmt(FW.q[n.idx], 4) + " model units\n= " + sgn(rem[n.idx], 3) + "% of the rest of the hour\n" + (NET.head_rule || "");
  }
  function hit(mx, my) {
    var best = null, bd = 1e9, i;
    for (i = 0; i < nodes.length; i++) { var n = nodes[i], d = Math.hypot(n.x - mx, n.y - my); if (d < n.r + 4 && d < bd) { bd = d; best = n; } }
    if (best) return best;
    for (i = 0; i < edges.length; i++) { var e = edges[i]; if (e.kind === "out") continue; var d2 = Math.hypot(e.mx - mx, e.my - my); if (d2 < 5 && d2 < bd) { bd = d2; best = e; } }
    return best;
  }
  var hoverPend = null;
  $("brain").addEventListener("mousemove", function (ev) {
    var r = $("brain").getBoundingClientRect();
    hoverPend = [ev.clientX - r.left, ev.clientY - r.top];
    requestAnimationFrame(function () {
      if (!hoverPend) return;
      var it = hit(hoverPend[0], hoverPend[1]), tip = $("tip");
      if (it !== hoverItem) { hoverItem = it; drawBrain(performance.now() / 1000); }
      if (!it) { tip.style.display = "none"; return; }
      tip.textContent = tipFor(it);
      tip.style.display = "block";
      var x = hoverPend[0] + 14, y = hoverPend[1] + 10, wr = $("brain-wrap").clientWidth;
      if (x + 380 > wr) x = Math.max(0, hoverPend[0] - 394);
      tip.style.left = x + "px"; tip.style.top = y + "px";
      hoverPend = null;
    });
  });
  $("brain").addEventListener("mouseleave", function () { hoverItem = null; $("tip").style.display = "none"; drawBrain(performance.now() / 1000); });

  // ------------------------------------------------------------------ pause / replay
  $("pause").addEventListener("click", function () {
    paused = !paused; replay = null;
    $("pause").textContent = paused ? "Resume live" : "Pause"; $("pause").setAttribute("aria-pressed", String(paused));
    if (!paused) { PREV.replay = null; kick(); }
  });
  $("replay").addEventListener("click", function () {
    if (frames.length < 2) return;
    paused = false; $("pause").textContent = "Pause";
    var t0 = Math.max(frames[0].t, nowMs() - WIN_S * 1000);
    replay = { t: t0, k: frames.findIndex(function (f) { return f.t >= t0; }), speed: +$("speed").value };
    PREV.replay = null; $("replay").textContent = "Replaying..."; kick();
  });
  function stepReplay(dt) {
    replay.t += dt * 1000 * replay.speed;
    while (replay.k < frames.length && frames[replay.k].t <= replay.t) {
      var fr = frames[replay.k++], b = BASES[fr.line];
      if (b) compute(fr.t, fr.st, b, "replay", true);
    }
    if (replay.k >= frames.length) {
      replay = null; PREV.replay = null; $("replay").textContent = "Replay 15 min";
      if (PREV.live) { FW = PREV.live.fw; FW0 = C.forward(NET, BASE.inputs.map(function (v) { return v === null ? NaN : v; })); }
    }
  }

  // ------------------------------------------------------------------ data: base, klines, websocket
  function setBase(b) {
    var changed = !BASE || b.line_id !== BASE.line_id;
    BASE = b; BASES[b.line_id] = b;
    $("m-line").textContent = b.line_id + (b.current ? "" : " (hour closed)");
    var rs = b.recorder || {};
    $("m-rec").textContent = rs.running ? "on (" + (rs.steps || 0) + " steps)" : "off";
    $("state-note").textContent = b.current ? "" : "The base line's hour has closed; waiting for the hh:03 line of the new hour.";
    if (changed) {
      ST = { P: b.p0, H: b.p0, L: b.p0, t_last_ms: b.S_ms, n: 0 };
      priceSec = []; PREV.live = null;
      FW0 = C.forward(NET, b.inputs.map(function (v) { return v === null ? NaN : v; }));
      if (!nodes.length) { build(); layout(); }
      return getJSON("/api/kline?hours=1.5").then(function (k) {
        var nm = nowMs(), rowsK = (k.rows || []).filter(function (r) { return r[0] >= b.S_ms && r[0] < b.S_ms + C.HOUR_MS; });
        var st = C.partialState(b.p0, b.S_ms, nm, [], rowsK);  // closed candles only
        rowsK.forEach(function (r) { if (r[0] + 60000 > nm) { st.H = Math.max(st.H, r[2]); st.L = Math.min(st.L, r[3]); st.P = r[4]; st.t_last_ms = nm; } });
        ST = { P: Math.max(st.P, 0) || b.p0, H: Math.max(st.H, ST.H), L: Math.min(st.L, ST.L), t_last_ms: Math.max(st.t_last_ms, ST.t_last_ms), n: st.n + ST.n };
        rowsK.forEach(function (r) { priceSec.push([Math.min(r[0] + 60000, nm), r[4]]); });
        dirty = true;
      }).catch(function () { dirty = true; });
    }
    return Promise.resolve();
  }
  function pollBase() { return getJSON("/api/brain/base").then(setBase).catch(function (e) { $("state-note").textContent = "Base unavailable: " + e.message; }); }

  function onPrice(ts, p) {
    if (!(p > 0) || !BASE) return;
    var off = ts - Date.now();
    clockOff = clockOff * 0.9 + off * 0.1;
    if (ts < BASE.S_ms || ts > BASE.S_ms + C.HOUR_MS) { $("px").textContent = p.toFixed(1); return; }
    ST.P = p; if (p > ST.H) ST.H = p; if (p < ST.L) ST.L = p; ST.t_last_ms = Math.max(ST.t_last_ms, ts); ST.n++;
    lastTickTs = ts; dirty = true;
    $("px").textContent = p.toFixed(1);
    $("px-chg").textContent = sgn(100 * Math.log(p / BASE.p0), 3) + "% this hour";
    var sec = Math.floor(ts / 1000) * 1000, lp = priceSec[priceSec.length - 1];
    if (lp && lp[0] >= sec) lp[1] = p; else priceSec.push([sec, p]);
    if (priceSec.length > 4000) priceSec.splice(0, priceSec.length - 4000);
  }

  var ws = null, pingT = null, backoff = 1000, lastMsg = 0;
  function setFeed(cls, txt) { $("ws-dot").className = "dot " + cls; $("ws-txt").textContent = txt; }
  function connect() {
    setFeed("wait", "connecting");
    try { ws = new WebSocket(WS_URL); } catch (e) { setFeed("bad", "websocket blocked"); setTimeout(connect, backoff); return; }
    ws.onopen = function () {
      ws.send(JSON.stringify({ op: "subscribe", args: ["publicTrade.BTCUSDT", "tickers.BTCUSDT"] }));
      clearInterval(pingT); pingT = setInterval(function () { if (ws.readyState === 1) ws.send(JSON.stringify({ op: "ping" })); }, PING_MS);
      backoff = 1000; lastMsg = Date.now(); setFeed("ok", "live");
    };
    ws.onmessage = function (e) {
      lastMsg = Date.now();
      var m; try { m = JSON.parse(e.data); } catch (x) { return; }
      if (m.topic === "publicTrade.BTCUSDT" && m.data) m.data.forEach(function (t) { onPrice(+t.T, +t.p); });
      else if (m.topic === "tickers.BTCUSDT" && m.data && m.data.lastPrice) onPrice(+m.ts, +m.data.lastPrice);
    };
    ws.onclose = function () { clearInterval(pingT); setFeed("bad", "reconnecting"); setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 30000); };
    ws.onerror = function () { try { ws.close(); } catch (x) { /* closed */ } };
  }
  setInterval(function () { if (ws && ws.readyState === 1 && Date.now() - lastMsg > 30000) { setFeed("wait", "silent, reconnecting"); ws.close(); } }, 5000);

  // ------------------------------------------------------------------ loops
  function pump() {  // the throttled forward pass (<= 1 / COMPUTE_MS), at least once a second (f moves with time)
    if (!NET || !BASE || !ST || document.hidden) return;
    var t = nowMs(), pt = performance.now();
    if (!BASE.current || t >= BASE.S_ms + C.HOUR_MS) return;
    if (!dirty && pt - lastCompute < IDLE_MS) return;
    dirty = false; lastCompute = pt;
    var st = { P: ST.P, H: ST.H, L: ST.L, t_last_ms: ST.t_last_ms, n: ST.n };
    compute(t, st, BASE, "live", !paused && !replay);
    frames.push({ t: t, st: st, line: BASE.line_id, x: Array.from(FW.x) });
    while (frames.length && frames[0].t < t - WIN_S * 1000) frames.shift();
    $("m-f").textContent = (100 * INFO.f).toFixed(1) + "%" + (INFO.warm ? " warm-up" : "");
    if (reduced.matches && !paused) drawBrain(performance.now() / 1000);
  }
  var cpuT0 = performance.now();
  function timed(fn) { return function () { var a = performance.now(); fn(); cpuBusy += performance.now() - a; }; }
  setInterval(timed(pump), COMPUTE_MS);
  setInterval(timed(function () {
    if (document.hidden || !NET) return;
    var k = Math.exp(-1 / RATE_TAU); for (var i = 0; i < rate.length; i++) rate[i] *= k;
    drawRaster(); drawMeters(); drawPrice(); drawQ();
    if (performance.now() - lastWhy > 2000) { why(); lastWhy = performance.now(); }
    var el = (performance.now() - passT0) / 1000;
    if (el > 5) {
      $("m-hz").textContent = (passes / el).toFixed(1) + " /s";
      $("cpu").textContent = "page JS " + (100 * cpuBusy / (performance.now() - cpuT0)).toFixed(1) + "% of one core";
      passes = 0; passT0 = performance.now(); cpuBusy = 0; cpuT0 = performance.now();
    }
  }), 1000);
  setInterval(pollBase, BASE_POLL_MS);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) { PREV.live = null; pollBase(); kick(); } });
  window.addEventListener("resize", function () { layout(); kick(); });
  if (reduced.addEventListener) reduced.addEventListener("change", function () { spikes = []; kick(); });

  function legend() {
    $("legend").innerHTML = FAM_ORDER.map(function (f) { return "<span><i style='background:" + famColor(f) + "'></i>" + f + "</span>"; }).join("") +
      "<span><i style='background:" + colors["--pos"] + "'></i>edge output / change up</span><span><i style='background:" + colors["--neg"] + "'></i>down</span>" +
      "<span>solid ring = changes intra-hour; dashed = frozen until the close</span>";
  }

  function boot() {
    var t = store("btcmon.theme"), qt = new URLSearchParams(location.search).get("theme");
    if (qt && THEMES.indexOf(qt) >= 0) t = qt;
    readColors();
    if (t && THEMES.indexOf(t) >= 0) applyTheme(t);
    getJSON("/api/brain/net").then(function (n) {
      NET = n; US = n.unit_stats || null;
      $("brain-sub").textContent = "refit " + n.refit_month + ", " + n.seeds.length + " seeds, " + n.features.length + " inputs, pt " + n.pt_sha256;
      return pollBase();
    }).then(function () {
      if (!nodes.length) build();
      layout(); legend(); connect(); kick();
    }).catch(function (e) { $("why").innerHTML = "<li>Could not load the network: " + e.message + "</li>"; });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
