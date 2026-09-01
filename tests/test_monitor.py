"""
Tests for Monitor
Run with: python -m pytest tests/test_monitor.py -v
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.monitor import Monitor

# FIXED: Aligned structure data dictionary with Two-Stage metrics
SAMPLE_PAPERS = [
    {
        "arxiv_id": "2501.11111", 
        "title": "VLMs for Healthcare", 
        "final_score": 87.5,
        "semantic_score": 82.0, 
        "llm_score": 90.0, 
        "published": "2025-05-28", 
        "url": "https://arxiv.org/abs/2501.11111",
        "llm_reason": "Matches your interests."
    },
    {
        "arxiv_id": "2501.22222", 
        "title": "Agentic Robotics", 
        "final_score": 74.0,
        "semantic_score": 74.0, 
        "llm_score": 0.0, 
        "published": "2025-05-27", 
        "url": "https://arxiv.org/abs/2501.22222",
        "llm_reason": ""
    },
]


def make_monitor(tmp_path):
    log_file = os.path.join(tmp_path, "test_runs.jsonl")
    return Monitor(log_path=log_file)


def _read_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


class TestMonitor:

    def test_log_file_created(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_run_start(papers_fetched=50)
        assert os.path.exists(m.log_path)

    def test_log_run_start_writes_event(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_run_start(papers_fetched=42)
        events = _read_events(m.log_path)
        assert any(e["event"] == "run_start" for e in events)
        assert any(e.get("papers_fetched") == 42 for e in events)

    def test_log_recommendations_writes_one_per_paper(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_recommendations(SAMPLE_PAPERS)
        events = _read_events(m.log_path)
        recs = [e for e in events if e["event"] == "recommendation"]
        assert len(recs) == 2

    def test_log_recommendations_correct_rank(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_recommendations(SAMPLE_PAPERS)
        events = _read_events(m.log_path)
        recs = [e for e in events if e["event"] == "recommendation"]
        assert recs[0]["rank"] == 1
        assert recs[1]["rank"] == 2

    def test_log_recommendations_has_required_fields(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_recommendations(SAMPLE_PAPERS)
        events = _read_events(m.log_path)
        rec = next(e for e in events if e["event"] == "recommendation")
        
        # FIXED: Verifying new fine-grained keys exist in log outputs
        for field in ["arxiv_id", "title", "final_score", "semantic_score", "llm_score", "url", "rank", "llm_reason"]:
            assert field in rec, f"Missing field: {field}"

    def test_log_run_end_success(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_run_end(success=True, email_sent=True)
        events = _read_events(m.log_path)
        end = next(e for e in events if e["event"] == "run_end")
        assert end["success"] is True
        assert end["email_sent"] is True

    def test_log_run_end_failure(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_run_end(success=False)
        events = _read_events(m.log_path)
        end = next(e for e in events if e["event"] == "run_end")
        assert end["success"] is False

    def test_log_error_writes_message(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_error("ArXiv request failed")
        events = _read_events(m.log_path)
        err = next(e for e in events if e["event"] == "error")
        assert "ArXiv request failed" in err["message"]

    def test_all_events_have_timestamp(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_run_start(50)
        m.log_recommendations(SAMPLE_PAPERS)
        m.log_run_end(success=True)
        for e in _read_events(m.log_path):
            assert "timestamp" in e

    def test_all_events_have_run_id(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.log_run_start(50)
        m.log_recommendations(SAMPLE_PAPERS)
        for e in _read_events(m.log_path):
            assert "run_id" in e

    def test_multiple_runs_append(self, tmp_path):
        m1 = make_monitor(str(tmp_path))
        m1.log_run_start(10)
        m2 = Monitor(log_path=m1.log_path)
        m2.log_run_start(20)
        events = _read_events(m1.log_path)
        starts = [e for e in events if e["event"] == "run_start"]
        assert len(starts) == 2

    def test_print_stats_no_crash_on_empty(self, tmp_path):
        m = make_monitor(str(tmp_path))
        m.print_stats()  # should not raise

    def test_print_stats_shows_run_count(self, tmp_path, capsys):
        m = make_monitor(str(tmp_path))
        m.log_run_start(50)
        m.log_recommendations(SAMPLE_PAPERS)
        m.log_run_end(success=True)
        m.print_stats()
        out = capsys.readouterr().out
        assert "1" in out   # 1 run
        assert "2" in out   # 2 recommendations