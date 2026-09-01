"""
Agent 1: ArXiv Source Monitor
------------------------------
Fetches today's papers from ArXiv matching your research interests.
NO API KEY needed — ArXiv is completely free and open.

Usage:
    from src.agent_arxiv import ArXivMonitor

    monitor = ArXivMonitor()
    papers = monitor.fetch(max_results=50)
    # returns list of dicts ready to feed into RAGRanker
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timezone


# ArXiv categories most relevant to VLM + agents + healthcare
DEFAULT_CATEGORIES = ["cs.CV", "cs.AI", "cs.LG", "cs.CL", "cs.RO"]

# Your core keywords — edit these to match your interests!
DEFAULT_KEYWORDS = [
    "vision language model",
    "multimodal",
    "VLM",
    "AI agent",
    "agentic",
    "healthcare",
    "medical imaging",
    "document understanding",
    "embodied AI",
    "robotic",
]

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}


class ArXivMonitor:
    """
    Fetches recent ArXiv papers matching your keywords.

    Args:
        keywords:   list of search terms (OR'd together)
        categories: list of ArXiv categories to search in
        days_back:  how many days back to look (default 1 = today only)
    """

    def __init__(
        self,
        keywords: list[str] = DEFAULT_KEYWORDS,
        categories: list[str] = DEFAULT_CATEGORIES,
        days_back: int = 1,
    ):
        self.keywords = keywords
        self.categories = categories
        self.days_back = days_back

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, max_results: int = 30) -> list[dict]:
        """
        Fetch recent papers. Returns list of paper dicts:
        {
            "arxiv_id": str,
            "title":    str,
            "abstract": str,
            "authors":  list[str],
            "url":      str,
            "published": str,   # ISO date string
            "categories": list[str],
        }
        """
        query = self._build_query()
        print(f"🔍 Fetching ArXiv papers...")
        print(f"   Keywords : {', '.join(self.keywords[:4])}{'...' if len(self.keywords) > 4 else ''}")
        print(f"   Categories: {', '.join(self.categories)}")

        papers = self._query_arxiv(query, max_results)

        print(f"✅ Found {len(papers)} papers\n")
        return papers

    def fetch_by_ids(self, arxiv_ids: list[str]) -> list[dict]:
        """Fetch specific papers by ArXiv ID (e.g. '2401.12345')."""
        id_list = ",".join(arxiv_ids)
        url = f"{ARXIV_API}?id_list={id_list}"
        return self._parse_response(self._get(url))
# ── Query builder (Fixed for URL encoding) ─────────────────────────────────

    def _build_query(self) -> str:
        """
        Build ArXiv search query string without manual pre-encoding.
        Example output: (ti:"vision language model" OR abs:"VLM") AND (cat:cs.CV)
        """
        kw_parts = []
        for kw in self.keywords:
            kw_parts.append(f'ti:"{kw}"')  # Exact phrase matching in title
        for kw in self.keywords[:5]:
            kw_parts.append(f'abs:"{kw}"') # Exact phrase matching in abstract

        kw_query = "(" + " OR ".join(kw_parts) + ")"

        # Category part
        cat_parts = [f"cat:{c}" for c in self.categories]
        cat_query = "(" + " OR ".join(cat_parts) + ")"

        return f"{kw_query} AND {cat_query}"

    # ── HTTP + XML ────────────────────────────────────────────────────────────

    def _query_arxiv(self, query: str, max_results: int) -> list[dict]:
        # urlencode will safely handle the spaces, quotes, ANDs, and ORs completely.
        params = urllib.parse.urlencode({
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{ARXIV_API}?{params}"
        xml_text = self._get(url)
        return self._parse_response(xml_text)

    def _get(self, url: str) -> str:
        """Simple HTTP GET with retry."""
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "ResearchRadar/1.0"}
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return resp.read().decode("utf-8")
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"ArXiv request failed: {e}") from e
                time.sleep(2 ** attempt)  # 1s, 2s backoff

    def _parse_response(self, xml_text: str) -> list[dict]:
        root = ET.fromstring(xml_text)
        papers = []

        for entry in root.findall("atom:entry", NS):
            try:
                arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]
                title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
                published = entry.find("atom:published", NS).text[:10]  # YYYY-MM-DD

                authors = [
                    a.find("atom:name", NS).text
                    for a in entry.findall("atom:author", NS)
                ]

                categories = [
                    t.get("term")
                    for t in entry.findall("atom:category", NS)
                    if t.get("term")
                ]

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "published": published,
                    "categories": categories,
                })
            except Exception:
                continue  # skip malformed entries

        return papers