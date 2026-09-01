"""
Tests for Agent 1: ArXiv Monitor
Run with: python -m pytest tests/test_arxiv.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch , MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent_arxiv import ArXivMonitor


# Minimal valid ArXiv API XML response (2 entries)
MOCK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.11111v1</id>
    <title>Vision Language Models for Medical Imaging</title>
    <summary>We present a VLM approach for radiology report generation and clinical document understanding.</summary>
    <published>2025-01-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.CV"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.22222v2</id>
    <title>Agentic LLMs with Tool Use</title>
    <summary>An agent framework for autonomous task planning using language models and external tools.</summary>
    <published>2025-01-14T00:00:00Z</published>
    <author><name>Carol White</name></author>
    <category term="cs.AI"/>
  </entry>
</feed>"""


class TestArXivMonitor:

    def _monitor(self, **kwargs):
        return ArXivMonitor(**kwargs)

    # ── Query building ────────────────────────────────────────────────────────

    def test_query_contains_keywords(self):
        """Verifies that the query string contains the unencoded raw keyword values."""
        m = self._monitor(keywords=["vision language", "multimodal"])
        query = m._build_query()
        # Adjusted to expect the raw space syntax since encoding happens in _query_arxiv
        assert 'ti:"vision language"' in query
        assert 'ti:"multimodal"' in query

    def test_query_contains_categories(self):
        m = self._monitor(categories=["cs.CV", "cs.AI"])
        query = m._build_query()
        assert "cs.CV" in query
        assert "cs.AI" in query

    def test_query_is_non_empty(self):
        m = self._monitor()
        assert len(m._build_query()) > 10

    # ── XML parsing ───────────────────────────────────────────────────────────

    def test_parse_returns_correct_count(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert len(papers) == 2

    def test_parse_extracts_title(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert papers[0]["title"] == "Vision Language Models for Medical Imaging"

    def test_parse_extracts_abstract(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert "VLM" in papers[0]["abstract"]

    def test_parse_extracts_arxiv_id(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert papers[0]["arxiv_id"] == "2501.11111v1"

    def test_parse_extracts_authors(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert "Alice Smith" in papers[0]["authors"]
        assert "Bob Jones" in papers[0]["authors"]

    def test_parse_extracts_url(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert papers[0]["url"] == "https://arxiv.org/abs/2501.11111v1"

    def test_parse_extracts_published_date(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert papers[0]["published"] == "2025-01-15"

    def test_parse_extracts_categories(self):
        m = self._monitor()
        papers = m._parse_response(MOCK_XML)
        assert "cs.CV" in papers[0]["categories"]

    def test_parse_handles_empty_feed(self):
        empty_xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        m = self._monitor()
        papers = m._parse_response(empty_xml)
        assert papers == []

    def test_parse_skips_malformed_entry(self):
        bad_xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry><id>http://arxiv.org/abs/9999.00001v1</id></entry>
        </feed>"""
        m = self._monitor()
        papers = m._parse_response(bad_xml)
        assert isinstance(papers, list)

    # ── fetch() with mocked HTTP ──────────────────────────────────────────────

    def test_fetch_calls_arxiv_and_returns_papers(self):
        m = self._monitor()
        with patch.object(m, "_get", return_value=MOCK_XML):
            papers = m.fetch(max_results=10)
        assert len(papers) == 2
        assert papers[0]["title"] == "Vision Language Models for Medical Imaging"

    def test_fetch_output_has_required_fields(self):
        m = self._monitor()
        with patch.object(m, "_get", return_value=MOCK_XML):
            papers = m.fetch()
        required = {"arxiv_id", "title", "abstract", "authors", "url", "published"}
        for paper in papers:
            assert required.issubset(paper.keys()), f"Missing fields: {required - paper.keys()}"

    @patch("time.sleep", return_value=None)  # Fast-forward retries
    def test_fetch_retries_on_failure(self, mock_sleep):
        """Verifies that _get internally handles retries up to 3 times before working."""
        m = self._monitor()
        
        # Mocking urlopen to fail twice, then return the real mock response
        mock_response = MagicMock()
        mock_response.read.return_value = MOCK_XML.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        with patch("urllib.request.urlopen", side_effect=[Exception("fail1"), Exception("fail2"), mock_response]):
            res = m._get("https://export.arxiv.org/api/query?test")
            assert "Vision Language Models" in res

    def test_network_failure_raises_runtime_error(self):
        m = self._monitor()
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            with pytest.raises(RuntimeError, match="ArXiv request failed"):
                m._get("https://export.arxiv.org/api/query?test")