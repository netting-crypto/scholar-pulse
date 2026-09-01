"""
Monitoring Logger
------------------
Logs every pipeline run and paper recommendation to logs/runs.jsonl.
Each line is one JSON event — easy to analyse later.

This gives you:
- Full audit trail of every recommendation
- Data to measure precision over time
- Evidence for your course evaluation section

Log file: logs/runs.jsonl
Each line looks like:
{
  "event":      "recommendation",
  "run_id":     "2026-05-28T08:00:00",
  "rank":       1,
  "arxiv_id":   "2605.12345v1",
  "title":      "...",
  "final_score": 87.5,
  "rag_score":   0.71,
  "published":  "2026-05-28",
  "url":        "https://arxiv.org/abs/...",
  "timestamp":  "2026-05-28T08:00:05"
}
"""

import json
import os
from datetime import datetime, timezone


LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "runs.jsonl")


class Monitor:
    """
    Logs pipeline runs and recommendations.

    Usage:
        monitor = Monitor()
        monitor.log_run_start(papers_fetched=50)
        monitor.log_recommendations(top_papers)
        monitor.log_run_end(success=True)
    """

    def __init__(self, log_path: str = LOG_PATH):
        self.log_path = os.path.abspath(log_path)
        self.run_id   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def log_run_start(self, papers_fetched: int) -> None:
        self._write({
            "event":          "run_start",
            "run_id":         self.run_id,
            "papers_fetched": papers_fetched,
        })
        print(f"📝 Logging to {self.log_path}")

    def log_recommendations(self, papers: list) -> None:
        """Log each recommended paper as a separate event."""
        for rank, p in enumerate(papers, 1):
            self._write({
                "event":       "recommendation",
                "run_id":      self.run_id,
                "rank":        rank,
                "arxiv_id":    p.get("arxiv_id", ""),
                "title":       p.get("title", ""),
                "final_score":    p.get("final_score", 0),
                "semantic_score": p.get("semantic_score", 0),
                "llm_score":      p.get("llm_score", 0),
                "llm_reason":     p.get("llm_reason", ""),
                "published":      p.get("published", ""),
                "url":            p.get("url", ""),
                "explanation":    p.get("explanation", ""),
            })

    def log_run_end(self, success: bool, email_sent: bool = False) -> None:
        self._write({
            "event":      "run_end",
            "run_id":     self.run_id,
            "success":    success,
            "email_sent": email_sent,
        })

    def log_error(self, error: str) -> None:
        self._write({
            "event":   "error",
            "run_id":  self.run_id,
            "message": error,
        })

    # ── Stats helper ──────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        """Print a summary of all logged runs."""
        if not os.path.exists(self.log_path):
            print("No logs yet.")
            return

        runs, recs, errors = 0, 0, 0
        scores = []

        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev["event"] == "run_start":
                        runs += 1
                    elif ev["event"] == "recommendation":
                        recs += 1
                        scores.append(ev.get("final_score", 0))
                    elif ev["event"] == "error":
                        errors += 1
                except Exception:
                    continue

        avg = sum(scores) / len(scores) if scores else 0
        print(f"\n📊 Monitor Stats")
        print(f"   Total runs          : {runs}")
        print(f"   Total recommendations: {recs}")
        print(f"   Avg relevance score : {avg:.1f}/100")
        print(f"   Errors logged       : {errors}")
        print(f"   Log file            : {self.log_path}\n")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(self, event: dict) -> None:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
