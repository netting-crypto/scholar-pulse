"""
digest_sender.py

Agent 3: Digest Sender
-----------------------
Sends your daily digest email via the Resend Python library.
Optimized to map cleanly to RAGRanker's two-stage metrics payload output.
Includes ⭐–⭐⭐⭐⭐⭐ rating buttons + 🔖 Save for Later that POST back to the feedback server.
"""

import os
import urllib.parse
from datetime import datetime

try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False


FEEDBACK_BASE_URL = os.environ.get("FEEDBACK_BASE_URL", "http://localhost:8000")


class DigestSender:

    def __init__(self, api_key="", to_address="", dry_run=False):
        self.api_key    = api_key    or os.environ.get("RESEND_API_KEY", "")
        self.to_address = to_address or os.environ.get("DIGEST_TO", "")
        self.dry_run    = dry_run

    @staticmethod
    def _esc(text) -> str:
        """Safely casts to string and escapes characters preventing HTML injection breaks."""
        if text is None:
            return ""
        return (str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    # ------------------------------------------------------------------
    # Feedback helpers
    # ------------------------------------------------------------------

    # Pill-style text buttons — repeated emoji glyphs break in most email clients
    _STARS = [
        (1, "1★",     "#e05c5c", "Not relevant"),
        (2, "2★★",   "#BA7517", "Slightly relevant"),
        (3, "🔖",     "#5b8dd9", "Save for later"),
        (4, "4★★★★", "#1D9E75", "Very relevant"),
        (5, "5★★★★★", "#0f6e50", "Must read"),
    ]

    def _feedback_url(self, paper: dict, rating: int) -> str:
        """Build a feedback URL matching ground_truth_papers.csv columns."""
        # why_you_care = highlight tags joined (e.g. "#agents, #VLM, #reasoning")
        highlights = paper.get("llm_highlights", [])
        why = ", ".join(f"#{h.strip()}" for h in highlights if h.strip())
        params = urllib.parse.urlencode({
            "arxiv_id":    paper.get("arxiv_id", ""),
            "title":       str(paper.get("title", ""))[:120],
            "abstract":    str(paper.get("abstract", ""))[:300],
            "your_rating": rating,
            "why_you_care": why[:200],
        })
        return f"{FEEDBACK_BASE_URL}/feedback?{params}"

    def _feedback_buttons_html(self, paper: dict) -> str:
        """Return the ⭐–⭐⭐⭐⭐⭐ + 🔖 HTML block for a paper card."""
        buttons = ""
        for rating, label, color, title in self._STARS:
            url = self._esc(self._feedback_url(paper, rating))
            margin = "margin-left:10px;" if rating == 3 else ""
            buttons += (
                f'<a href="{url}" title="{title}" '
                f'style="text-decoration:none;font-size:12px;font-weight:600;'
                f'color:{color};border:1px solid {color};border-radius:4px;'
                f'padding:2px 8px;{margin}">'
                f'{label}</a>'
            )
        return f"""
        <div style="margin-top:12px;padding-top:10px;border-top:1px solid #e8e6de;
                    display:flex;align-items:center;gap:6px;font-size:12px;
                    color:#888780;flex-wrap:wrap;">
          <span style="margin-right:4px;">Rate:</span>
          {buttons}
        </div>"""

    def _feedback_buttons_text(self, paper: dict) -> str:
        """Return plain-text rating lines for the text fallback."""
        lines = ["   Rate this paper:"]
        for rating, label, color, title in self._STARS:
            url = self._feedback_url(paper, rating)
            lines.append(f"   {label} — {title}: {url}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, papers: list, stats: dict = None) -> bool:
        if not papers:
            print("No papers to send.")
            return False

        stats   = stats or {}
        subject = self._subject(stats)
        html    = self._build_html(papers, stats)
        text    = self._build_text(papers, stats)

        if not self.api_key and not self.dry_run:
            print("Missing RESEND_API_KEY. Defaulting to Dry Run print mode.")
            self._print_dry_run(subject, text)
            return False

        if self.dry_run:
            self._print_dry_run(subject, text)
            return True

        return self._send_resend(subject, html, text)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _subject(self, stats: dict) -> str:
        date  = stats.get("date", datetime.now().strftime("%b %d"))
        count = stats.get("total_fetched", "?")
        return f"ScholarPulse — {date} ({count} papers scanned)"

    def _build_text(self, papers: list, stats: dict) -> str:
        lines = [
            "SCHOLARPULSE — Daily Digest",
            "=" * 40,
            f"Date   : {stats.get('date', datetime.now().strftime('%Y-%m-%d'))}",
            f"Scanned: {stats.get('total_fetched', '?')} papers",
            "",
        ]
        for i, p in enumerate(papers, 1):
            authors    = p.get("authors", [])
            author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")

            reason = p.get("llm_reason", "")
            if not reason:
                reason = f"Semantic alignment score: {p.get('semantic_score', 0)}"

            lines += [
                f"{i}. {p['title']}",
                f"   Score  : {p['final_score']}/100",
                f"   Authors: {author_str}",
                f"   URL    : {p.get('url', '')}",
                f"   Why    : {reason}",
                self._feedback_buttons_text(p),  # 👍/👎 plain-text links
                "",
            ]
        return "\n".join(lines)

    def _build_html(self, papers: list, stats: dict) -> str:
        date_str   = stats.get("date", datetime.now().strftime("%B %d, %Y"))
        total      = stats.get("total_fetched", "?")
        match_rate = stats.get("match_rate", "")

        cards_html = ""
        for i, p in enumerate(papers, 1):
            score       = p.get("final_score", 0)
            title       = self._esc(p.get("title", ""))
            url         = self._esc(p.get("url", "#"))
            authors     = p.get("authors", [])
            author_str  = self._esc(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
            published   = self._esc(p.get("published", ""))

            why         = self._esc(p.get("llm_reason", ""))
            highlights  = p.get("llm_highlights", [])

            score_color = "#1D9E75" if score >= 70 else ("#BA7517" if score >= 50 else "#888780")

            tags_html = ""
            if highlights:
                tags_html = '<div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap;">'
                for h in highlights:
                    if h.strip():
                        tags_html += (
                            f'<span style="background:#e8e6de;color:#5f5e5a;font-size:11px;'
                            f'font-weight:600;padding:2px 8px;border-radius:4px;margin-right:4px;">'
                            f'#{self._esc(h)}</span>'
                        )
                tags_html += '</div>'

            why_html = (
                f'<div style="margin-top:10px;font-size:13px;color:#3d3d3a;line-height:1.5;">'
                f'<strong style="color:#5F5E5A;">Why it matters:</strong> {why}'
                f'{tags_html}'
                f'</div>'
            ) if why else ""

            # 👍/👎 buttons
            feedback_html = self._feedback_buttons_html(p)

            cards_html += f"""
            <div style="background:#ffffff;border:1px solid #e8e6de;border-radius:10px;
                        padding:20px 24px;margin-bottom:16px;">
              <div style="display:flex;align-items:flex-start;gap:14px;">
                <div style="font-size:22px;font-weight:700;color:#c8c4bc;min-width:28px;">{i}</div>
                <div style="flex:1">
                  <a href="{url}" style="font-size:15px;font-weight:600;color:#1a1a18;text-decoration:none;">{title}</a>
                  <div style="margin-top:6px;font-size:12px;color:#888780;">{author_str}</div>
                  <div style="margin-top:4px;font-size:12px;color:#888780;">{published} · {self._esc(p.get('arxiv_id', ''))}</div>
                  {why_html}
                  {feedback_html}
                </div>
                <div style="text-align:center;min-width:52px;">
                  <div style="font-size:20px;font-weight:700;color:{score_color};">{score}</div>
                  <div style="font-size:10px;color:#888780;">/ 100</div>
                </div>
              </div>
            </div>"""

        match_html = (
            f'<div><div style="font-size:18px;font-weight:700;color:#ffffff;">{match_rate}</div>'
            f'<div style="font-size:11px;color:#888780;">match rate</div></div>'
        ) if match_rate else ""

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f4f2eb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="max-width:620px;margin:32px auto;padding:0 16px;">
    <div style="background:#2C2C2A;border-radius:12px 12px 0 0;padding:28px 32px;">
      <div style="font-size:11px;font-weight:600;letter-spacing:1.5px;color:#888780;text-transform:uppercase;margin-bottom:6px;">Daily Digest</div>
      <div style="font-size:24px;font-weight:700;color:#ffffff;">ScholarPulse</div>
      <div style="font-size:13px;color:#B4B2A9;margin-top:4px;">{date_str}</div>
    </div>
    <div style="background:#3d3d3a;padding:12px 32px;display:flex;gap:32px;flex-wrap:wrap;">
      <div>
        <div style="font-size:18px;font-weight:700;color:#ffffff;">{total}</div>
        <div style="font-size:11px;color:#888780;">papers scanned</div>
      </div>
      <div>
        <div style="font-size:18px;font-weight:700;color:#ffffff;">{len(papers)}</div>
        <div style="font-size:11px;color:#888780;">top matches</div>
      </div>
      {match_html}
    </div>
    <div style="background:#f4f2eb;padding:24px 0;">
      <div style="font-size:11px;font-weight:600;letter-spacing:1px;color:#888780;text-transform:uppercase;margin-bottom:14px;">Top papers for you</div>
      {cards_html}
    </div>
    <div style="padding:16px 0 32px;text-align:center;font-size:11px;color:#B4B2A9;">
      ScholarPulse · Powered by ArXiv + Two-Stage RAG · ⭐ ratings improve your profile
    </div>
  </div>
</body>
</html>"""

    def _send_resend(self, subject: str, html: str, text: str) -> bool:
        if not RESEND_AVAILABLE:
            print("resend library not installed. Run: uv add resend")
            return False
        if not self.to_address:
            print("Missing DIGEST_TO environment configuration variable.")
            return False

        try:
            resend.api_key = self.api_key
            print(f"Sending digest to {self.to_address} ...")
            r = resend.Emails.send({
                "from":    "ScholarPulse <onboarding@resend.dev>",
                "to":      [self.to_address],
                "subject": subject,
                "html":    html,
                "text":    text,
            })
            print(f"Digest sent! (id: {r.get('id', 'ok')})")
            return True
        except Exception as e:
            print(f"Failed to send: {e}")
            return False

    def _print_dry_run(self, subject: str, text: str) -> None:
        print("\n" + "=" * 60)
        print("   DRY RUN — Local Content Delivery Dump")
        print("=" * 60)
        print(f"   To     : {self.to_address or '(set DIGEST_TO)'}")
        print(f"   Subject: {subject}")
        print("=" * 60)
        print(text)
        print("=" * 60 + "\n")
