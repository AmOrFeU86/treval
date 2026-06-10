"""Local web dashboard for treval.

Starts an HTTP server that displays spans in the browser.

Usage:
    treval dashboard            # Port 8080
    treval dashboard --port 3000
"""
from __future__ import annotations

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from treval.db import SpanStore


def _build_html(store=None):
    """Generates dashboard HTML with embedded data."""
    if store is None:
        store = SpanStore()
    spans = store.list_spans(limit=200)
    total = store.count()
    stats = {
        "total": total,
        "AGENT": sum(1 for s in spans if s["type"] == "AGENT"),
        "OPERATION": sum(1 for s in spans if s["type"] == "OPERATION"),
        "TOOL": sum(1 for s in spans if s["type"] == "TOOL"),
        "LLM": sum(1 for s in spans if s["type"] == "LLM"),
        "errors": sum(1 for s in spans if s["status"] == "error"),
    }
    data = json.dumps({"spans": spans, "stats": stats}, default=str)
    return HTML_TEMPLATE.replace("__DATA__", data)


class DashboardHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the dashboard."""

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self._send_html()
        elif self.path.startswith("/api/spans"):
            self._send_spans()
        elif self.path.startswith("/api/span/"):
            span_id = int(self.path.split("/")[-1])
            self._send_span(span_id)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_build_html().encode())

    def _send_spans(self):
        store = SpanStore()
        spans = store.list_spans(limit=200)
        total = store.count()
        stats = {
            "total": total,
            "AGENT": sum(1 for s in spans if s["type"] == "AGENT"),
            "OPERATION": sum(1 for s in spans if s["type"] == "OPERATION"),
            "TOOL": sum(1 for s in spans if s["type"] == "TOOL"),
            "LLM": sum(1 for s in spans if s["type"] == "LLM"),
            "errors": sum(1 for s in spans if s["status"] == "error"),
        }
        data = {
            "spans": [{k: v for k, v in s.items() if k != "metadata"} for s in spans],
            "stats": stats,
        }
        self._send_json(data)

    def _send_span(self, span_id: int):
        store = SpanStore()
        span = store.get(span_id)
        if not span:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return
        children = store.get_children(span_id)
        span["children"] = children
        self._send_json(dict(span))

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args):
        pass  # Silent


def serve(port: int = 8080, open_browser: bool = True):
    """Starts the dashboard server."""
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"treval dashboard at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>treval dashboard</title>
<link rel="icon" type="image/x-icon" href="data:image/x-icon;base64,AAABAAEAICAAAAAAIADgBAAAFgAAAIlQTkcNChoKAAAADUlIRFIAAAAgAAAAIAgCAAAA/BjtowAABKdJREFUeJx9VktMXFUY/v5zzmVmgOHNKAhWKViEkBKp1QYlxmjaULDWnfGx0cSNcaOufSyMK000MXHFUk00bhpXldqNLSkm0kojgikxVNQGC8zwutxzfxf33HvPuUM9mZncM+fc//H93/8gmR8AE6IvQAAAhlkUP7C9dS+ZtwjMzivRkSIQI7souydi6xZbcgFjGZsHVxqJSDcdJDbVxXzAn6k4iqyMVBIoVkwAhKXRoHSQobZuc4viD1fbHdsFQMTbxF1znR0d9rK3FMNOieEWeABYue8wkA1sldBqZZHEFMco3pGcxANwRnXqaep0oDnQFqZAHADHAst7EgA4Vk6W1xm2MVjr8PGh7qdH7teaY1hs56qoBwBQAGKmRmo44Q2lVwkAEd26XfaUJEc7bPMjb+xjUc3ByIlMMoW+j1DvNPdWmh9AGIS+b5lJsZokpSimCYsoPlEqJLTL2A4pG86Oo8arKzYUG5tYyYaz41DKwAymmIcU408uRCnWlAiNQ87M2t+jja2G9u7luVmAmtq7aXMr8HclCYKIYkRxfrhhIJGFxwYGYA5zhWLf0NjGhYtBZbO7/9ihgePBzvb69MXewcdyhSJzGBmRRCI21/ggjGfG4BQ7ZgT7ATN7uXzPkUdq69sq/642tXY0t3eW11YK9a09/Se8XJ6ZARJSpkanvwyAVG7QJT0zwMw1nnekv2dubgGBDxJ1hbsDbIcBSSGYtCeLlcoKpACopqbm7bde+ejjKd/fd/kFBikT4wQSgAAdBE+eHPvwgzffee/T4aP9N26sbGxujZ4YufbL757yzp//8XBfp+f1PXxsqK216bPPv+zsLGW4m5Qs4eiL80tIOT+/dOnyz1evLvQePnRpZu6N119cXFyamBhtaaudPDN6enxsfaM8NfXNue9+eOH5iZ2dXXZEG6AoYlGkimHzgMrlcrFYVyq1rv516+bK6vr6hvTU9PTMF1+d+/brT76/MHNPR+nR40crle3m5sYwDK3YOsJI5QZgigmRnXTMz555qrK1s367PDMzOzw8Mnn61OxPc9PTl0+dfGLht4XlP5Zefuk5rcP564tSyitXrulQZyhugswWOCn9wXrPj8pwbV0zaU8DpdbefC6/+s+vrAMNf2drDSAICbDyVAI9W8kkYsyy5ZAAlasRUnT0PPjMa+8XS22qwIzdPb/Myq9ta5l89d2u/odICeUp5XlpHTDSDTNJ5QfBdnlwQGTmXKGusdS19udyqPfrC+1Eorz9txCqpeO+zbXVve0yEcHUOHtwMMKiPLCAd0qF+YtDLaRi1sXau4jExtaqIKF1IIQkElZIyW08QJwHpgklFcvubEQgqQAwc1NDlxByvXITJKX0rAyNDc+UZorywEhkjuQZ0jKD7VrPHLY23dvS2A0ObRfvsEw3IpUfjJ4yjcKKR7pC1iAIyOxFZzmIiUR6JNEdzpLmaowVJETitHMHbltOn9O5iExVIgtZuzTaPnEMYyKO4/6cHRyicp0ZuRJG21NJ9SlZW/vUmdXSwYvSY3Yv2cqqXHG2lo64zYiEMweMQtn1/7Rx1ZsRxVTTA61IhNKdtR74irUl/AfoQRBytDgiIQAAAABJRU5ErkJggg=="/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0b0e14; --surface: #14181f; --surface2: #1c2330;
  --border: #2a3348; --text: #e2e8f0; --text2: #8899b4;
  --accent: #60a5fa; --green: #34d399; --red: #f87171; --yellow: #fbbf24; --purple: #a78bfa;
  --radius: 10px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

/* Header */
.header { background: linear-gradient(135deg, #14181f 0%, #1c2330 100%); border-bottom: 1px solid var(--border); padding: 12px 16px; position: sticky; top: 0; z-index: 10; backdrop-filter: blur(12px); }
.header-inner { max-width: 1200px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px; }
.header h1 { font-size: 1.1rem; font-weight: 700; letter-spacing: -.02em; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.meta { font-size: .78rem; color: var(--text2); }

/* Main */
.main { max-width: 1200px; margin: 0 auto; padding: 16px; }

/* Stats */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; margin-bottom: 16px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 12px; text-align: center; cursor: help; transition: transform .15s, border-color .15s; }
.stat:hover { transform: translateY(-1px); border-color: var(--accent); }
.stat .num { font-size: 1.3rem; font-weight: 700; line-height: 1.2; }
.num.green { color: var(--green); } .num.blue { color: var(--accent); } .num.yellow { color: var(--yellow); } .num.purple { color: var(--purple); } .num.red { color: var(--red); }
.stat .label { font-size: .62rem; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; }

/* Table */
.table-wrap { overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface); }
table { width: 100%; border-collapse: collapse; min-width: 580px; }
thead { background: var(--surface2); }
th { padding: 10px 12px; text-align: left; font-size: .7rem; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; border-bottom: 1px solid var(--border); cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { color: var(--text); }
th .sort { opacity: 0; margin-left: 4px; }
th.sorted .sort { opacity: 1; }
td { padding: 8px 12px; border-top: 1px solid var(--border); font-size: .8rem; vertical-align: middle; }
tr { cursor: pointer; transition: background .1s; }
tr:hover { background: color-mix(in srgb, var(--accent) 6%, transparent); }
tr.active { background: color-mix(in srgb, var(--accent) 10%, transparent); }

/* Type badges */
.type-badge { display: inline-block; padding: 1px 8px; border-radius: 12px; font-size: .65rem; font-weight: 600; letter-spacing: .02em; cursor: help; }
.type-AGENT { background: color-mix(in srgb, var(--accent) 15%, transparent); color: var(--accent); }
.type-OPERATION { background: color-mix(in srgb, var(--green) 15%, transparent); color: var(--green); }
.type-TOOL { background: color-mix(in srgb, var(--yellow) 15%, transparent); color: var(--yellow); }
.type-LLM { background: color-mix(in srgb, var(--purple) 15%, transparent); color: var(--purple); }

/* Status dots */
.status-indicator { display: inline-flex; align-items: center; gap: 5px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-ok { background: var(--green); box-shadow: 0 0 6px rgba(52, 211, 153, .4); }
.dot-error { background: var(--red); box-shadow: 0 0 6px rgba(248, 113, 113, .4); }
.status-text { font-size: .75rem; font-weight: 500; }

/* Duration bar */
.duration-wrap { display: flex; align-items: center; gap: 6px; }
.duration-bar { height: 4px; border-radius: 2px; flex: 1; min-width: 30px; max-width: 60px; background: var(--border); overflow: hidden; }
.duration-bar-fill { height: 100%; border-radius: 2px; transition: width .3s; }
.duration-text { font-family: monospace; font-size: .75rem; color: var(--text2); white-space: nowrap; }
.id-cell { font-family: monospace; font-size: .75rem; color: var(--accent); }

/* Tree indent */
.tree-indent { display: inline-block; width: 0; margin-right: 4px; font-size: .6rem; color: var(--text2); opacity: .5; }

/* Detail panel */
.detail-panel { margin-top: 12px; animation: slideIn .2s ease; }
@keyframes slideIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
.detail-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; }
.detail-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.detail-header h2 { font-size: .95rem; font-weight: 600; }
.detail-header .close-btn { margin-left: auto; background: none; border: 1px solid var(--border); color: var(--text2); width: 24px; height: 24px; border-radius: 6px; cursor: pointer; font-size: .8rem; display: flex; align-items: center; justify-content: center; transition: background .15s; }
.detail-header .close-btn:hover { background: var(--border); }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 12px; }
.detail-field .fname { font-size: .65rem; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 3px; font-weight: 600; }
.detail-field .fval { font-size: .82rem; word-break: break-word; }
.detail-field .fval pre { background: var(--bg); padding: 8px 10px; border-radius: 6px; font-size: .75rem; line-height: 1.4; overflow-x: auto; white-space: pre-wrap; word-break: break-all; border: 1px solid var(--border); margin-top: 3px; max-height: 150px; overflow-y: auto; }
.children-section { border-top: 1px solid var(--border); padding-top: 12px; }
.children-section h3 { font-size: .75rem; color: var(--text2); margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.child-item { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: var(--bg); border-radius: 6px; margin-bottom: 4px; font-size: .78rem; }

/* Legend */
.legend { margin-top: 16px; padding: 12px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); font-size: .75rem; color: var(--text2); line-height: 1.6; }
.legend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 6px; margin-top: 6px; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.legend-item .type-badge { cursor: default; }
.legend-item span { font-size: .72rem; }

/* Animations */
tr { animation: rowIn .2s ease both; }
@keyframes rowIn { from { opacity: 0; transform: translateX(-4px); } to { opacity: 1; transform: translateX(0); } }

/* Mobile */
@media (max-width: 640px) {
  .header { padding: 8px 10px; }
  .header-inner { flex-direction: row; align-items: center; gap: 4px; }
  .header h1 { font-size: .9rem; }
  .meta { font-size: .68rem; }
  .main { padding: 8px; }
  .stats { grid-template-columns: repeat(3, 1fr); gap: 4px; margin-bottom: 8px; }
  .stat { padding: 6px 4px; border-radius: 8px; }
  .stat .num { font-size: .95rem; }
  .stat .label { font-size: .55rem; }
  table { min-width: 0; }
  thead { display: none; }
  tr { display: block; padding: 8px 10px; border-top: 1px solid var(--border); animation: none; }
  td { display: flex; justify-content: space-between; align-items: center; padding: 3px 0; border: none; font-size: .72rem; }
  td::before { content: attr(data-label); font-weight: 600; color: var(--text2); font-size: .6rem; text-transform: uppercase; letter-spacing: .04em; margin-right: 8px; flex-shrink: 0; }
  td:last-child { border: none; }
  .table-wrap { border: none; background: transparent; }
  tr:hover { background: var(--surface); }
  .duration-bar { display: none; }
  .detail-card { padding: 10px 12px; }
  .detail-grid { grid-template-columns: 1fr; gap: 6px; }
  .detail-header { gap: 6px; margin-bottom: 8px; }
  .detail-header h2 { font-size: .82rem; }
  .child-item { font-size: .72rem; padding: 4px 8px; gap: 4px; }
  .legend { display: none; }
}
@media (max-width: 380px) {
  .stats { grid-template-columns: repeat(2, 1fr); }
  .header h1 { font-size: .8rem; }
}
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="header">
<div class="header-inner">
<h1>&#x26A1; treval dashboard</h1>
<div class="meta" id="meta">Loading... <span id="meta-detail"></span></div>
</div>
</div>
<div class="main">
<div class="stats" id="stats"></div>
<div class="table-wrap">
<table>
<thead><tr>
<th onclick="sortBy('id')">ID<span class="sort">&#x25B2;</span></th>
<th onclick="sortBy('type')">Type<span class="sort">&#x25B2;</span></th>
<th onclick="sortBy('name')">Name<span class="sort">&#x25B2;</span></th>
<th onclick="sortBy('status')">Status<span class="sort">&#x25B2;</span></th>
<th onclick="sortBy('duration_ms')">Duration<span class="sort">&#x25B2;</span></th>
<th>Parent</th>
<th onclick="sortBy('created_at')">Started<span class="sort">&#x25B2;</span></th>
</tr></thead>
<tbody id="spans-tbody"></tbody>
</table>
</div>
<div class="detail-panel" id="detail-panel"></div>
<div class="legend" id="legend">
<div style="font-weight: 600; margin-bottom: 4px;">&#x2139;&#xFE0F; Span hierarchy</div>
<div class="legend-grid">
<div class="legend-item"><span class="type-badge type-AGENT">AGENT</span> <span>Full AI agent</span></div>
<div class="legend-item"><span class="type-badge type-OPERATION">OPERATION</span> <span>Operation inside agent</span></div>
<div class="legend-item"><span class="type-badge type-TOOL">TOOL</span> <span>Tool or function executed</span></div>
<div class="legend-item"><span class="type-badge type-LLM">LLM</span> <span>LLM call</span></div>
<div class="legend-item"><span class="status-indicator"><span class="status-dot dot-ok"></span></span> <span>Success</span></div>
<div class="legend-item"><span class="status-indicator"><span class="status-dot dot-error"></span></span> <span>Error</span></div>
</div>
</div>
</div>

<script>
var _EMBEDDED = __DATA__;
var _SORT = { field: null, asc: true };

function load() {
  if (_EMBEDDED) { render(_EMBEDDED); }
  else { fetch("/api/spans?limit=200").then(function(r){return r.json()}).then(render); }
}

function render(data) {
  var n = data.spans.length;
  document.getElementById("meta").textContent = n + " span" + (n !== 1 ? "s" : "") + " / " + data.stats.total + " total";
  var cm = {total:"blue",AGENT:"blue",OPERATION:"green",TOOL:"yellow",LLM:"purple",errors:"red"};
  var lm = {total:"Total",AGENT:"Agents",OPERATION:"Operations",TOOL:"Tools",LLM:"LLMs",errors:"Errors"};
  var h = "";
  Object.keys(data.stats).forEach(function(k) {
    h += '<div class="stat" title="' + (lm[k]||k) + ': ' + data.stats[k] + '"><div class="num ' + (cm[k]||"blue") + '">' + data.stats[k] + '</div><div class="label">' + (lm[k]||k) + '</div></div>';
  });
  document.getElementById("stats").innerHTML = h;
  _DATA = data.spans;
  renderTable();
}

var _DATA = [];

function renderTable() {
  var tb = document.getElementById("spans-tbody");
  tb.innerHTML = "";
  var items = _DATA;
  if (_SORT.field) {
    var f = _SORT.field;
    var m = _SORT.asc ? 1 : -1;
    items = items.slice().sort(function(a,b) {
      var va = a[f], vb = b[f];
      if (va == null) va = "";
      if (vb == null) vb = "";
      if (typeof va === "number") return (va - vb) * m;
      return String(va).localeCompare(String(vb)) * m;
    });
  }
  var maxDur = 0;
  items.forEach(function(s) { if (s.duration_ms && s.duration_ms > maxDur) maxDur = s.duration_ms; });
  var delay = 0;
  items.forEach(function(s) {
    var tr = document.createElement("tr");
    tr.setAttribute("data-id", s.id);
    tr.onclick = function(){toggleDetail(s.id)};
    tr.style.animationDelay = (delay * 0.02) + "s";
    delay++;

    var dur = s.duration_ms != null ? s.duration_ms.toFixed(1) : null;
    var durText = dur != null ? dur + "ms" : "&mdash;";
    var durPct = dur != null && maxDur > 0 ? Math.max(3, (dur / maxDur) * 100) : 0;
    var durColor = dur != null && dur > 50 ? "var(--yellow)" : dur != null && dur > 200 ? "var(--red)" : "var(--accent)";
    var cre = (s.created_at || "").slice(0, 19);
    var dotClass = s.status === "error" ? "dot-error" : "dot-ok";
    var depth = s.parent_id ? 1 : 0;

    tr.innerHTML =
      '<td data-label="ID"><span class="id-cell">#' + s.id + '</span></td>' +
      '<td data-label="Type"><span class="type-badge type-' + s.type + '" title="' + typeHint(s.type) + '">' + s.type + '</span></td>' +
      '<td data-label="Name"><span class="tree-indent">' + (depth ? '&#x2514;&#x2500;' : '') + '</span>' + esc(s.name) + '</td>' +
      '<td data-label="Status"><span class="status-indicator"><span class="status-dot ' + dotClass + '"></span><span class="status-text" style="color:' + (s.status==="error"?"var(--red)":"var(--green)") + '">' + s.status + '</span></span></td>' +
      '<td data-label="Duration"><span class="duration-wrap"><span class="duration-bar"><span class="duration-bar-fill" style="width:' + durPct + '%;background:' + durColor + '"></span></span><span class="duration-text">' + durText + '</span></span></td>' +
      '<td data-label="Parent">' + (s.parent_id || "&mdash;") + '</td>' +
      '<td data-label="Started">' + cre + '</td>';
    tb.appendChild(tr);
  });
}

function typeHint(t) {
  return {AGENT:"Full AI agent",OPERATION:"Operation inside agent",TOOL:"Tool or function executed",LLM:"LLM call"}[t] || "";
}

function sortBy(field) {
  if (_SORT.field === field) { _SORT.asc = !_SORT.asc; }
  else { _SORT.field = field; _SORT.asc = true; }
  document.querySelectorAll("th .sort").forEach(function(s){s.style.opacity="0"});
  document.querySelectorAll("th").forEach(function(t){t.classList.remove("sorted")});
  var ths = document.querySelectorAll("th");
  var idx = {id:0,type:1,name:2,status:3,duration_ms:4,created_at:6}[field];
  if (ths[idx]) { ths[idx].classList.add("sorted"); ths[idx].querySelector(".sort").style.opacity = "1"; }
  if (_SORT.asc) { ths[idx].querySelector(".sort").innerHTML = "&#x25B2;"; }
  else { ths[idx].querySelector(".sort").innerHTML = "&#x25BC;"; }
  renderTable();
}

function toggleDetail(id) {
  var panel = document.getElementById("detail-panel");
  var prev = panel.getAttribute("data-span-id");
  document.querySelectorAll("#spans-tbody tr.active").forEach(function(r){r.classList.remove("active")});
  if (prev == id) { panel.innerHTML = ""; panel.removeAttribute("data-span-id"); return; }
  if (_EMBEDDED) {
    var s = _EMBEDDED.spans.find(function(sp){return sp.id === id}) || {};
    s.children = _EMBEDDED.spans.filter(function(sp){return sp.parent_id === id});
    showDetail(panel, s, id);
  } else {
    fetch("/api/span/" + id).then(function(r){return r.json()}).then(function(s){showDetail(panel, s, id)});
  }
}

function showDetail(panel, s, id) {
  panel.setAttribute("data-span-id", id);
  var row = document.querySelector('#spans-tbody tr[data-id="' + id + '"]');
  if (row) row.classList.add("active");
  var dur = s.duration_ms != null ? s.duration_ms.toFixed(1) + "ms" : "&mdash;";
  var ch = "";
  if (s.children && s.children.length) {
    ch = '<div class="children-section"><h3>Children (' + s.children.length + ')</h3>';
    s.children.forEach(function(c){
      ch += '<div class="child-item"><span class="type-badge type-' + c.type + '">' + c.type + '</span> <strong>' + esc(c.name) + '</strong> <span class="status-indicator"><span class="status-dot ' + (c.status==="error"?"dot-error":"dot-ok") + '"></span></span> <span class="duration-text">' + (c.duration_ms != null ? c.duration_ms.toFixed(1) + "ms" : "&mdash;") + '</span></div>';
    });
    ch += "</div>";
  }
  var dotClass = s.status === "error" ? "dot-error" : "dot-ok";
  var closeId = 'close-detail-' + id;
  panel.innerHTML =
    '<div class="detail-card">' +
      '<div class="detail-header"><span class="type-badge type-' + s.type + '">' + s.type + '</span><h2>' + esc(s.name||"") + '</h2><span class="status-indicator"><span class="status-dot ' + dotClass + '"></span></span><span style="font-size:.78rem;color:var(--text2)">#' + s.id + '</span><button id="' + closeId + '" class="close-btn">&times;</button></div>' +
      '<div class="detail-grid">' +
        '<div class="detail-field"><div class="fname">Status</div><div class="fval"><span class="status-indicator"><span class="status-dot ' + dotClass + '"></span> <span style="color:' + (s.status==="error"?"var(--red)":"var(--green)") + '">' + s.status + '</span></span></div></div>' +
        '<div class="detail-field"><div class="fname">Duration</div><div class="fval">' + dur + '</div></div>' +
        '<div class="detail-field"><div class="fname">Parent</div><div class="fval">' + (s.parent_id || "&mdash;") + '</div></div>' +
        '<div class="detail-field"><div class="fname">Started</div><div class="fval">' + (s.created_at || "&mdash;") + '</div></div>' +
      '</div>' +
      '<div class="detail-grid">' +
        '<div class="detail-field"><div class="fname">Input</div><div class="fval"><pre>' + esc(s.input||"&mdash;") + '</pre></div></div>' +
        '<div class="detail-field"><div class="fname">Output</div><div class="fval"><pre>' + esc(s.output||"&mdash;") + '</pre></div></div>' +
      '</div>' +
      ch +
    '</div>';
  setTimeout(function() {
    var btn = document.getElementById(closeId);
    if (btn) btn.onclick = function() {
      panel.innerHTML = "";
      panel.removeAttribute("data-span-id");
    };
  }, 0);
  setTimeout(function(){ panel.scrollIntoView({behavior:"smooth",block:"start"}); }, 50);
}

function esc(str) {
  if (!str) return "";
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

load();
</script>
</body>
</html>
"""