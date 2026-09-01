"""
Tests for Agent 3: Digest Sender (Resend)
Run with: python -m pytest tests/test_digest.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# FIXED: Import cleanly from digest_sender file path. Removed global _esc import.
from src.digest_sender import DigestSender

# UPDATED CONTRACT: Converted 'explanation' keys to 'llm_reason' and added 'llm_highlights'
SAMPLE_PAPERS = [
    {
        "title": "Vision Language Models for Medical Imaging",
        "final_score": 87.5,
        "authors": ["Alice Smith", "Bob Jones", "Carol White"],
        "published": "2025-05-28",
        "url": "https://arxiv.org/abs/2501.11111",
        "categories": ["cs.CV", "cs.AI"],
        "llm_reason": "Directly combines VLMs with healthcare imaging.",
        "llm_highlights": ["VLM", "Healthcare"]
    },
    {
        "title": "Agentic LLMs with Tool Use for Robotics",
        "final_score": 74.0,
        "authors": ["Dan Brown"],
        "published": "2025-05-27",
        "url": "https://arxiv.org/abs/2501.22222",
        "categories": ["cs.RO", "cs.AI"],
        "llm_reason": "Agents + robotics, two of your core interests.",
        "llm_highlights": ["Agents", "Robotics"]
    },
    {
        "title": "Document Layout Understanding with Multimodal Transformers",
        "final_score": 61.0,
        "authors": [],
        "published": "2025-05-26",
        "url": "https://arxiv.org/abs/2501.33333",
        "categories": ["cs.CV"],
        "llm_reason": "",
        "llm_highlights": []
    },
]

SAMPLE_STATS = {
    "date": "May 28, 2025",
    "total_fetched": 47,
    "match_rate": "6.4%",
}


class TestEscapeHelper:
    """
    UPDATED: Pointing to static methods.
    Why: _esc is now part of the DigestSender class schema block, requiring 
    calls via DigestSender._esc() instead of global scope evaluation.
    """
    def test_escapes_ampersand(self):
        assert DigestSender._esc("AT&T") == "AT&amp;T"

    def test_escapes_lt_gt(self):
        assert DigestSender._esc("<b>bold</b>") == "&lt;b&gt;bold&lt;/b&gt;"

    def test_escapes_quotes(self):
        assert DigestSender._esc('say "hi"') == "say &quot;hi&quot;"

    def test_plain_text_unchanged(self):
        assert DigestSender._esc("hello world") == "hello world"

    def test_handles_none_safely(self):
        """[ADDED FEATURE TEST] Verifies that None type inputs fall back to empty string."""
        assert DigestSender._esc(None) == ""


class TestDigestSender:

    def _sender(self, **kwargs):
        return DigestSender(dry_run=True, **kwargs)

    def test_subject_contains_date(self):
        assert "May 28" in self._sender()._subject({"date": "May 28"})

    def test_subject_contains_paper_count(self):
        assert "47" in self._sender()._subject({"total_fetched": 47})

    def test_subject_works_without_stats(self):
        assert len(self._sender()._subject({})) > 5

    def test_text_contains_all_titles(self):
        text = self._sender()._build_text(SAMPLE_PAPERS, SAMPLE_STATS)
        for p in SAMPLE_PAPERS:
            assert p["title"] in text

    def test_text_contains_scores(self):
        assert "87.5" in self._sender()._build_text(SAMPLE_PAPERS, SAMPLE_STATS)

    def test_text_contains_urls(self):
        assert "https://arxiv.org/abs/2501.11111" in self._sender()._build_text(SAMPLE_PAPERS, SAMPLE_STATS)

    def test_text_truncates_long_author_list(self):
        assert "et al." in self._sender()._build_text(SAMPLE_PAPERS, SAMPLE_STATS)

    def test_html_is_valid_structure(self):
        html = self._sender()._build_html(SAMPLE_PAPERS, SAMPLE_STATS)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_html_contains_all_titles(self):
        html = self._sender()._build_html(SAMPLE_PAPERS, SAMPLE_STATS)
        for p in SAMPLE_PAPERS:
            assert p["title"] in html

    def test_html_contains_arxiv_links(self):
        assert "https://arxiv.org/abs/2501.11111" in self._sender()._build_html(SAMPLE_PAPERS, SAMPLE_STATS)

    def test_html_shows_stats(self):
        html = self._sender()._build_html(SAMPLE_PAPERS, SAMPLE_STATS)
        assert "47" in html
        assert "6.4%" in html

    def test_html_escapes_special_chars(self):
        papers = [{**SAMPLE_PAPERS[0], "title": "VLMs & <Healthcare>"}]
        html = self._sender()._build_html(papers, {})
        assert "<Healthcare>" not in html
        assert "&lt;Healthcare&gt;" in html

    def test_html_renders_highlight_tags(self):
        """[ADDED FEATURE TEST] Assures that markdown hashtag styling arrays parse smoothly."""
        html = self._sender()._build_html(SAMPLE_PAPERS, SAMPLE_STATS)
        assert "#VLM" in html
        assert "#Healthcare" in html

    def test_send_returns_true_on_dry_run(self):
        assert self._sender().send(SAMPLE_PAPERS, SAMPLE_STATS) is True

    def test_send_returns_false_on_empty_papers(self):
        assert self._sender().send([], SAMPLE_STATS) is False

    def test_send_prints_output_in_dry_run(self, capsys):
        self._sender().send(SAMPLE_PAPERS[:1], {})
        assert "DRY RUN" in capsys.readouterr().out

    def test_missing_api_key_returns_false(self):
        """
        UPDATED: Confirms that a production call with missing API values falls back 
        gracefully by showing the dry-run console print block before exiting.
        """
        s = DigestSender(api_key="", to_address="x@x.com", dry_run=False)
        assert s.send(SAMPLE_PAPERS, SAMPLE_STATS) is False

    def test_missing_to_address_returns_false(self):
        s = DigestSender(api_key="re_fake", to_address="", dry_run=False)
        assert s.send(SAMPLE_PAPERS, SAMPLE_STATS) is False

    def test_credentials_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("RESEND_API_KEY", "re_testkey")
        monkeypatch.setenv("DIGEST_TO", "me@example.com")
        s = DigestSender()
        assert s.api_key    == "re_testkey"
        assert s.to_address == "me@example.com"