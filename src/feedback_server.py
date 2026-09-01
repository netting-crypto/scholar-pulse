"""
feedback_server.py

Writes ratings back into the existing ground_truth_papers.csv format:
  arxiv_id | title | abstract | your_rating | why_you_care

Accepts both old param names (paper_id, rating) and new ones (arxiv_id, your_rating)
so emails sent before the update still work.
"""

import csv
import datetime
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="ScholarPulse Feedback")

FEEDBACK_CSV = os.environ.get("FEEDBACK_CSV", "data/ground_truth_papers.csv")
FIELDNAMES = ["arxiv_id", "title", "abstract", "your_rating", "why_you_care", "updated_at"]

RATING_META = {
    1: ("1★",      "Not relevant",      "#e05c5c"),
    2: ("2★★",    "Slightly relevant",  "#BA7517"),
    3: ("3★★★",   "Neutral / save",     "#888780"),
    4: ("4★★★★",  "Very relevant",      "#1D9E75"),
    5: ("5★★★★★", "Must read",         "#0f6e50"),
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_rows() -> list:
    if not os.path.exists(FEEDBACK_CSV):
        return []
    with open(FEEDBACK_CSV, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _write_rows(rows: list) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_CSV) or ".", exist_ok=True)
    with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _upsert(arxiv_id: str, title: str, abstract: str,
            your_rating: int, why_you_care: str) -> bool:
    """
    Update existing row if arxiv_id already in CSV, else append.
    Returns True if it was a new entry.
    """
    rows  = _read_rows()
    found = False

    for row in rows:
        if row.get("arxiv_id", "").strip() == arxiv_id.strip():
            row["your_rating"]  = your_rating
            row["why_you_care"] = why_you_care or row.get("why_you_care", "")
            row["updated_at"]   = datetime.datetime.utcnow().isoformat()
            if not row.get("abstract") and abstract:
                row["abstract"] = abstract
            found = True
            break

    if not found:
        rows.append({
            "arxiv_id":    arxiv_id,
            "title":       title,
            "abstract":    abstract,
            "your_rating": your_rating,
            "why_you_care": why_you_care,
            "updated_at":  datetime.datetime.utcnow().isoformat()
        })

    _write_rows(rows)
    return not found


def _rebuild_profile() -> None:
    try:
        import importlib, sys
        sys.path.insert(0, ".")
        mod    = importlib.import_module("src.rag_ranker")
        ranker = mod.RAGRanker()
        ranker.build_interest_profile()
        print("Interest profile rebuilt.")
    except Exception as e:
        print(f"Could not rebuild profile: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/feedback", response_class=HTMLResponse)
def receive_feedback(
    # new param names (emails sent after the update)
    arxiv_id:     str = Query(""),
    your_rating:  int = Query(0, ge=0, le=5),
    # old param names (emails sent before the update)
    paper_id:     str = Query(""),
    rating:       int = Query(0, ge=0, le=5),
    # shared params
    title:        str = Query(""),
    abstract:     str = Query(""),
    why_you_care: str = Query(""),
):
    # normalise: prefer new names, fall back to old
    resolved_id     = arxiv_id    or paper_id
    resolved_rating = your_rating or rating

    if not resolved_id or not resolved_rating:
        raise HTTPException(status_code=422, detail="arxiv_id and rating are required")

    is_new = _upsert(resolved_id, title, abstract, resolved_rating, why_you_care)
    _rebuild_profile()

    emoji, label, color = RATING_META[resolved_rating]
    action_word = "Added to" if is_new else "Updated in"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ScholarPulse — Rated</title>
</head>
<body style="margin:0;padding:0;background:#f4f2eb;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="max-width:480px;margin:60px auto;padding:0 24px;text-align:center;">
    <div style="font-size:48px;margin-bottom:12px;">{emoji}</div>
    <h2 style="color:{color};margin:0 0 10px;">{label}</h2>
    <p style="font-size:14px;color:#3d3d3a;margin:0 0 4px;">
      <strong>{title[:80]}{"..." if len(title) > 80 else ""}</strong>
    </p>
    <p style="font-size:12px;color:#888780;margin:0 0 20px;">{resolved_id}</p>
    <hr style="border:none;border-top:1px solid #e8e6de;margin:0 0 20px;">
    <p style="font-size:13px;color:#3d3d3a;">
      {action_word} <code>ground_truth_papers.csv</code>
      with rating <strong>{resolved_rating}/5</strong>.
    </p>
    <p style="font-size:12px;color:#888780;margin-top:8px;">
      Your interest profile has been updated.
    </p>
  </div>
</body>
</html>"""


@app.get("/stats")
def stats():
    rows = _read_rows()
    by_rating = {}
    for r in rows:
        k = str(r.get("your_rating", "?"))
        by_rating[k] = by_rating.get(k, 0) + 1
    return JSONResponse({"total": len(rows), "by_rating": by_rating, "rows": rows})


@app.get("/health")
def health():
    return {"status": "ok"}
