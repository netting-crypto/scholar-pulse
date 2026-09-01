"""
ScholarPulse — Web Dashboard
--------------------------------
Reads logs/runs.jsonl and data/ground_truth_papers.csv and serves a live
dashboard at http://localhost:5000

Run:
    uv run python dashboard.py
    # then open http://localhost:5000 in your browser

No extra dependencies — uses Python's built-in http.server.
"""

import json
import os
import csv
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from collections import defaultdict

LOG_PATH     = os.path.join(os.path.dirname(__file__), "logs", "runs.jsonl")
FEEDBACK_CSV = os.path.join(os.path.dirname(__file__), "data", "ground_truth_papers.csv")
PORT = 5000

RATING_LABELS = {
    "1": "⭐ Not relevant",
    "2": "⭐⭐ Slightly relevant",
    "3": "🔖 Saved for later",
    "4": "⭐⭐⭐⭐ Very relevant",
    "5": "⭐⭐⭐⭐⭐ Must read",
}
RATING_COLORS = {
    "1": "#e05c5c",
    "2": "#BA7517",
    "3": "#5b8dd9",
    "4": "#1D9E75",
    "5": "#0f6e50",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_stats():
    if not os.path.exists(LOG_PATH):
        return {"runs": [], "total_runs": 0, "total_recs": 0,
                "avg_score": 0, "recent_papers": [], "scores_over_time": [], "errors": 0}

    runs, recs, errors = {}, [], 0
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev  = json.loads(line)
                rid = ev.get("run_id", "")
                if ev["event"] == "run_start":
                    runs[rid] = {"run_id": rid, "papers_fetched": ev.get("papers_fetched", 0),
                                 "timestamp": ev.get("timestamp", ""), "recommendations": []}
                elif ev["event"] == "recommendation":
                    recs.append(ev)
                    if rid in runs:
                        runs[rid]["recommendations"].append(ev)
                elif ev["event"] == "run_end":
                    if rid in runs:
                        runs[rid]["email_sent"] = ev.get("email_sent", False)
                        runs[rid]["success"]    = ev.get("success", False)
                elif ev["event"] == "error":
                    errors += 1
            except Exception:
                continue

    run_list   = sorted(runs.values(), key=lambda r: r["run_id"], reverse=True)
    all_scores = [r.get("final_score", 0) for r in recs]
    avg_score  = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    seen, recent = set(), []
    for rec in sorted(recs, key=lambda r: r.get("timestamp", ""), reverse=True):
        aid = rec.get("arxiv_id", rec.get("title", ""))
        if aid not in seen:
            seen.add(aid)
            recent.append(rec)
        if len(recent) >= 20:
            break

    scores_over_time = []
    for r in reversed(run_list[-14:]):
        rs = [x.get("final_score", 0) for x in r.get("recommendations", [])]
        scores_over_time.append({
            "run_id": r["run_id"][:10],
            "avg":    round(sum(rs) / len(rs), 1) if rs else 0,
            "count":  len(rs),
        })

    return {
        "total_runs"       : len(run_list),
        "total_recs"       : len(recs),
        "avg_score"        : avg_score,
        "errors"           : errors,
        "runs"             : run_list[:10],
        "recent_papers"    : recent,
        "scores_over_time" : scores_over_time,
    }


def load_ratings():
    if not os.path.exists(FEEDBACK_CSV):
        return None

    with open(FEEDBACK_CSV, newline="", encoding="cp1252", errors="replace") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    by_rating = defaultdict(int)
    by_date   = defaultdict(lambda: defaultdict(int))
    saved     = []

    for row in rows:
        r = str(row.get("your_rating", "") or row.get("rating", "")).strip()
        if not r or r == "0":
            continue
        by_rating[r] += 1
        updated = row.get("updated_at", "")[:10]
        if updated:
            by_date[updated][r] += 1
        # treat rating=3 as "saved for later" (no separate saved column in CSV)
        if r == "3":
            saved.append(row)

    saved.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

    all_dates  = sorted(by_date.keys())
    all_buckets = ["1", "2", "3", "4", "5"]
    timeline   = [
        {"date": d, **{k: by_date[d].get(k, 0) for k in all_buckets}}
        for d in all_dates
    ]

    # only keep rows that actually have a rating
    rated_rows = [r for r in rows if str(r.get("your_rating", "") or r.get("rating", "")).strip() not in ("", "0")]

    return {
        "total"      : len(rated_rows),
        "positive"   : sum(by_rating.get(k, 0) for k in ["4", "5"]),
        "negative"   : sum(by_rating.get(k, 0) for k in ["1", "2"]),
        "saved"      : saved,
        "saved_count": len(saved),
        "by_rating"  : dict(by_rating),
        "timeline"   : timeline,
        "rows"       : sorted(rated_rows, key=lambda x: x.get("updated_at", ""), reverse=True)[:30],
    }


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def render_papers_html(stats):
    html = ""
    for p in stats["recent_papers"]:
        score = p.get("final_score", 0)
        color = "#1D9E75" if score >= 70 else ("#BA7517" if score >= 50 else "#888780")
        title = p.get("title", "")[:80]
        url   = p.get("url", "#")
        date  = p.get("published", "")
        rank  = p.get("rank", "")
        html += f"""
        <div class="paper-card">
          <div class="paper-score" style="color:{color}">{score}</div>
          <div class="paper-body">
            <a href="{url}" target="_blank" class="paper-title">{title}</a>
            <div class="paper-meta">Run: {p.get('run_id','')[:10]} &nbsp;·&nbsp; {date} &nbsp;·&nbsp; Rank #{rank}</div>
          </div>
        </div>"""
    return html


def render_runs_html(stats):
    html = ""
    for r in stats["runs"]:
        ts      = r.get("timestamp", "")[:16].replace("T", " ")
        count   = len(r.get("recommendations", []))
        fetched = r.get("papers_fetched", 0)
        email   = "✓" if r.get("email_sent") else "–"
        rs      = [x.get("final_score", 0) for x in r.get("recommendations", [])]
        avg     = round(sum(rs) / len(rs), 1) if rs else 0
        html += f"""
        <tr>
          <td>{ts}</td><td>{fetched}</td><td>{count}</td>
          <td>{avg}/100</td><td>{email}</td>
        </tr>"""
    return html


def render_ratings_section(rt):
    if rt is None:
        return """
        <div class="section">
          <h2>Your ratings</h2>
          <p class="empty">No ratings yet — click ⭐ buttons in your digest emails to get started.</p>
        </div>"""

    # Distribution bars
    dist_html = ""
    for k in ["5", "4", "3", "2", "1"]:
        count = rt["by_rating"].get(k, 0)
        pct   = round(count / rt["total"] * 100) if rt["total"] else 0
        dist_html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <div style="width:160px;font-size:12px;color:#5F5E5A;white-space:nowrap;">{RATING_LABELS[k]}</div>
          <div style="flex:1;background:#f0ede6;border-radius:4px;height:14px;overflow:hidden;">
            <div style="width:{pct}%;background:{RATING_COLORS[k]};height:100%;border-radius:4px;"></div>
          </div>
          <div style="width:32px;text-align:right;font-size:12px;color:#888780;">{count}</div>
        </div>"""

    # Saved papers
    saved_html = ""
    if rt["saved"]:
        for row in rt["saved"][:10]:
            saved_html += f"""
            <div class="paper-card">
              <div class="paper-body">
                <span class="paper-title" style="color:#1a1a18;">{row['title'][:80]}</span>
                <div class="paper-meta">Saved {row.get('updated_at','')[:10]}</div>
              </div>
            </div>"""
    else:
        saved_html = '<p class="empty">No saved papers yet. Click 🔖 in any digest email.</p>'

    # Recent ratings table
    table_rows = ""
    for row in rt["rows"]:
        r     = str(row.get("your_rating", "") or row.get("rating", "")).strip()
        color = RATING_COLORS.get(r, "#888780")
        label = RATING_LABELS.get(r, r)
        date  = row.get("updated_at", "")[:10]
        title = row.get("title", "")[:70]
        table_rows += f"""
        <tr>
          <td style="max-width:340px;">{title}</td>
          <td><span style="color:{color};font-weight:600;">{label}</span></td>
          <td>{date}</td>
        </tr>"""

    # Timeline stacked bar chart
    tl          = rt["timeline"]
    tl_labels   = json.dumps([x["date"] for x in tl])
    tl_datasets = json.dumps([
        {"label": RATING_LABELS[k],
         "data":  [x[k] for x in tl],
         "backgroundColor": RATING_COLORS[k],
         "borderRadius": 3}
        for k in ["5", "4", "3", "2", "1"]
    ])

    chart_block = f"""
        <canvas id="ratingsChart" style="max-height:180px;margin-bottom:24px;"></canvas>
        <script>
        (function(){{
          new Chart(document.getElementById('ratingsChart'), {{
            type: 'bar',
            data: {{ labels: {tl_labels}, datasets: {tl_datasets} }},
            options: {{
              responsive: true,
              plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 11 }} }} }} }},
              scales: {{
                x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }},
                y: {{ stacked: true, grid: {{ color: '#f0ede6' }}, ticks: {{ font: {{ size: 11 }}, stepSize: 1 }} }}
              }}
            }}
          }});
        }})();
        </script>""" if tl else '<p class="empty" style="margin-bottom:20px;">Rate papers across multiple days to see the timeline.</p>'

    return f"""
    <div class="stats" style="grid-template-columns:repeat(3,1fr);margin-bottom:20px;">
      <div class="stat">
        <div class="stat-val">{rt['total']}</div>
        <div class="stat-label">Papers rated</div>
      </div>
      <div class="stat">
        <div class="stat-val" style="color:#1D9E75;">{rt['positive']}</div>
        <div class="stat-label">Highly relevant (4–5 ⭐)</div>
      </div>
      <div class="stat">
        <div class="stat-val" style="color:#5b8dd9;">{rt['saved_count']}</div>
        <div class="stat-label">Saved for later 🔖</div>
      </div>
    </div>

    <div class="section">
      <h2>Ratings over time</h2>
      {chart_block}
      <h2 style="margin-bottom:12px;">Distribution</h2>
      {dist_html}
    </div>

    <div class="section">
      <h2>🔖 Saved for later</h2>
      {saved_html}
    </div>

    <div class="section">
      <h2>Recent ratings</h2>
      {"<table><thead><tr><th>Paper</th><th>Rating</th><th>Date</th></tr></thead><tbody>" + table_rows + "</tbody></table>" if table_rows else '<p class="empty">No ratings yet.</p>'}
    </div>"""


def render_html(stats, rt):
    papers_html     = render_papers_html(stats)
    runs_html       = render_runs_html(stats)
    ratings_section = render_ratings_section(rt)

    chart_labels = json.dumps([s["run_id"] for s in stats["scores_over_time"]])
    chart_data   = json.dumps([s["avg"]    for s in stats["scores_over_time"]])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScholarPulse — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f4f2eb;color:#2C2C2A;font-size:14px}}
  .header{{background:#2C2C2A;color:#fff;padding:24px 32px;display:flex;align-items:center;gap:16px}}
  .header h1{{font-size:20px;font-weight:600}}
  .header .sub{{font-size:12px;color:#888780;margin-top:2px}}
  .tabs{{background:#3d3d3a;padding:0 32px;display:flex}}
  .tab{{color:#B4B2A9;font-size:13px;padding:12px 20px;cursor:pointer;border-bottom:2px solid transparent;user-select:none}}
  .tab:hover{{color:#fff}}
  .tab.active{{color:#fff;border-bottom-color:#534AB7}}
  .refresh{{margin-left:auto;background:#3d3d3a;border:none;color:#B4B2A9;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px}}
  .refresh:hover{{background:#4d4d4a}}
  .main{{max-width:960px;margin:0 auto;padding:28px 20px}}
  .page{{display:none}}.page.active{{display:block}}
  .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
  .stat{{background:#fff;border:1px solid #e8e6de;border-radius:10px;padding:18px 20px}}
  .stat-val{{font-size:28px;font-weight:700;color:#2C2C2A;line-height:1}}
  .stat-label{{font-size:11px;color:#888780;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}}
  .section{{background:#fff;border:1px solid #e8e6de;border-radius:10px;padding:20px 24px;margin-bottom:20px}}
  .section h2{{font-size:13px;font-weight:600;color:#5F5E5A;text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px}}
  .paper-card{{display:flex;align-items:flex-start;gap:14px;padding:12px 0;border-bottom:1px solid #f0ede6}}
  .paper-card:last-child{{border-bottom:none}}
  .paper-score{{font-size:18px;font-weight:700;min-width:40px;text-align:right;line-height:1.3}}
  .paper-body{{flex:1}}
  .paper-title{{font-size:13px;font-weight:500;color:#1a1a18;text-decoration:none;line-height:1.4}}
  .paper-title:hover{{color:#534AB7}}
  .paper-meta{{font-size:11px;color:#888780;margin-top:3px}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;font-size:11px;font-weight:600;color:#888780;text-transform:uppercase;letter-spacing:.5px;padding:0 0 8px;border-bottom:1px solid #e8e6de}}
  td{{padding:10px 0;border-bottom:1px solid #f4f2eb;font-size:12px;color:#5F5E5A}}
  tr:last-child td{{border-bottom:none}}
  .empty{{color:#B4B2A9;font-size:13px;padding:20px 0}}
  canvas{{max-height:160px}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>ScholarPulse</h1>
    <div class="sub">Last updated: {datetime.now().strftime("%B %d, %Y %H:%M")}</div>
  </div>
  <button class="refresh" onclick="location.reload()">Refresh</button>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('runs', this)">Pipeline runs</div>
  <div class="tab"        onclick="switchTab('ratings', this)">My ratings</div>
</div>

<div class="main">

  <!-- RUNS PAGE -->
  <div id="page-runs" class="page active">
    <div class="stats">
      <div class="stat"><div class="stat-val">{stats["total_runs"]}</div><div class="stat-label">Total runs</div></div>
      <div class="stat"><div class="stat-val">{stats["total_recs"]}</div><div class="stat-label">Papers ranked</div></div>
      <div class="stat"><div class="stat-val">{stats["avg_score"]}</div><div class="stat-label">Avg relevance score</div></div>
      <div class="stat"><div class="stat-val">{stats["errors"]}</div><div class="stat-label">Errors logged</div></div>
    </div>

    <div class="section">
      <h2>Avg relevance score per run</h2>
      {"<canvas id='scoreChart'></canvas>" if stats["scores_over_time"] else '<p class="empty">No runs yet — run uv run python pipeline.py to see data here.</p>'}
    </div>

    <div class="section">
      <h2>Recent recommendations</h2>
      {papers_html or '<p class="empty">No papers logged yet.</p>'}
    </div>

    <div class="section">
      <h2>Run history</h2>
      {"<table><thead><tr><th>Time</th><th>Fetched</th><th>Shown</th><th>Avg score</th><th>Email</th></tr></thead><tbody>" + runs_html + "</tbody></table>" if runs_html else '<p class="empty">No runs logged yet.</p>'}
    </div>
  </div>

  <!-- RATINGS PAGE -->
  <div id="page-ratings" class="page">
    {ratings_section}
  </div>

</div>

<script>
function switchTab(name, el) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  el.classList.add('active');
}}

const scoreLabels = {chart_labels};
const scoreData   = {chart_data};
if (scoreLabels.length > 0) {{
  new Chart(document.getElementById('scoreChart'), {{
    type: 'line',
    data: {{
      labels: scoreLabels,
      datasets: [{{
        data: scoreData,
        borderColor: '#534AB7',
        backgroundColor: 'rgba(83,74,183,0.08)',
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: '#534AB7',
        tension: 0.3,
        fill: true,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ min: 0, max: 100, grid: {{ color: '#f0ede6' }}, ticks: {{ font: {{ size: 11 }} }} }},
        x: {{ grid: {{ display: false }},  ticks: {{ font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server (zero extra deps — stdlib only)
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        stats = load_stats()
        rt    = load_ratings()
        html  = render_html(stats, rt).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print(f"Dashboard running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    HTTPServer(("", PORT), Handler).serve_forever()
