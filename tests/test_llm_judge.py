"""
tests/test_llm_judge.py

User Satisfaction Evaluation for ScholarPulse.
Computes average user rating of recommended papers from ground_truth_papers.csv.

This is the primary judge test: instead of asking an LLM to evaluate results,
we use REAL ratings you clicked in your digest emails to measure how well
the system is performing over time.

Run with:
    uv run pytest tests/test_llm_judge.py -v

No API key required — reads directly from data/ground_truth_papers.csv.
"""

import os
import sys
import csv
from datetime import datetime, timedelta
from collections import defaultdict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FEEDBACK_CSV  = os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth_papers.csv")
MIN_PAPERS    = 3      # minimum rated papers needed to run satisfaction tests
PASS_AVG      = 3.0    # minimum acceptable average rating (out of 5)
GOOD_AVG      = 4.0    # target average rating we aim for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rated_papers():
    """Load all papers that have a numeric your_rating from the CSV."""
    if not os.path.exists(FEEDBACK_CSV):
        return []

    rated = []
    encodings = ["utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            with open(FEEDBACK_CSV, newline="", encoding=enc, errors="replace") as f:
                for row in csv.DictReader(f):
                    try:
                        rating = float(row.get("your_rating", "") or 0)
                        if rating > 0:
                            rated.append({
                                "arxiv_id":    row.get("arxiv_id", ""),
                                "title":       row.get("title", ""),
                                "your_rating": rating,
                                "why_you_care": row.get("why_you_care", ""),
                                "updated_at":  row.get("updated_at", ""),
                            })
                    except (ValueError, TypeError):
                        continue
            break
        except Exception:
            continue

    return rated


def _avg(ratings):
    return round(sum(ratings) / len(ratings), 2) if ratings else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUserSatisfaction:

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.papers = _load_rated_papers()
        if len(self.papers) < MIN_PAPERS:
            pytest.skip(
                f"Only {len(self.papers)} rated papers found in ground_truth_papers.csv "
                f"(need at least {MIN_PAPERS}). Rate more papers from your digest emails."
            )

    # ------------------------------------------------------------------

    def test_average_rating_above_minimum(self):
        """
        Core satisfaction test: average rating of all recommended papers
        must be above the minimum acceptable threshold (3.0 / 5.0).

        A score below 3.0 means the system is recommending mostly irrelevant papers.
        """
        ratings = [p["your_rating"] for p in self.papers]
        avg     = _avg(ratings)
        total   = len(ratings)

        print(f"\n📊 User Satisfaction Score")
        print(f"   Total rated papers : {total}")
        print(f"   Average rating     : {avg:.2f} / 5.00")
        print(f"   Min acceptable     : {PASS_AVG:.1f} / 5.00")
        print(f"   Target             : {GOOD_AVG:.1f} / 5.00")

        assert avg >= PASS_AVG, (
            f"Average rating {avg:.2f}/5 is below minimum {PASS_AVG}/5. "
            f"The ranker needs tuning — too many irrelevant papers are being recommended."
        )

    def test_highly_relevant_ratio(self):
        """
        At least 40% of recommended papers should be rated 4 or 5
        (interesting or must-read).
        """
        ratings      = [p["your_rating"] for p in self.papers]
        high_ratings = [r for r in ratings if r >= 4]
        ratio        = len(high_ratings) / len(ratings)

        print(f"\n⭐ High-relevance ratio (rating 4-5)")
        print(f"   Papers rated 4-5 : {len(high_ratings)} / {len(ratings)}")
        print(f"   Ratio            : {ratio:.0%}")
        print(f"   Target           : ≥ 40%")

        assert ratio >= 0.40, (
            f"Only {ratio:.0%} of papers were rated 4-5 (target: ≥40%). "
            f"The ranker is not surfacing enough high-quality papers."
        )

    def test_irrelevant_ratio_is_low(self):
        """
        Fewer than 30% of recommended papers should be rated 1 or 2
        (not relevant or irrelevant).
        """
        ratings     = [p["your_rating"] for p in self.papers]
        low_ratings = [r for r in ratings if r <= 2]
        ratio       = len(low_ratings) / len(ratings)

        print(f"\n👎 Low-relevance ratio (rating 1-2)")
        print(f"   Papers rated 1-2 : {len(low_ratings)} / {len(ratings)}")
        print(f"   Ratio            : {ratio:.0%}")
        print(f"   Target           : < 30%")

        assert ratio < 0.30, (
            f"{ratio:.0%} of papers were rated 1-2 (target: <30%). "
            f"Too many irrelevant papers are reaching the digest."
        )

    def test_rating_distribution_is_reported(self):
        """
        Non-failing report: prints a full breakdown of ratings.
        Always passes — exists to show the distribution in pytest output.
        """
        dist = defaultdict(int)
        for p in self.papers:
            dist[int(p["your_rating"])] += 1

        total = len(self.papers)
        avg   = _avg([p["your_rating"] for p in self.papers])

        print(f"\n📈 Rating distribution ({total} papers total, avg {avg:.2f}/5)")
        labels = {
            5: "Must read     ⭐⭐⭐⭐⭐",
            4: "Interesting   ⭐⭐⭐⭐",
            3: "Maybe         ⭐⭐⭐",
            2: "Not relevant  ⭐⭐",
            1: "Irrelevant    ⭐",
        }
        for star in [5, 4, 3, 2, 1]:
            count = dist.get(star, 0)
            pct   = count / total * 100 if total else 0
            bar   = "█" * int(pct / 5)
            print(f"   {labels[star]} : {count:3d} ({pct:5.1f}%) {bar}")

        # Always passes — this is a reporting test
        assert True

    def test_recent_trend_is_stable_or_improving(self):
        """
        Compare average rating of the most recent 10 papers vs all-time average.
        Recent average should not be more than 0.5 points below all-time average
        (i.e. quality is not degrading over time).

        Skipped if fewer than 10 papers have timestamps.
        """
        timestamped = [
            p for p in self.papers if p.get("updated_at")
        ]
        if len(timestamped) < 10:
            pytest.skip("Need at least 10 timestamped ratings to check trend.")

        timestamped.sort(key=lambda x: x["updated_at"], reverse=True)
        recent      = timestamped[:10]
        recent_avg  = _avg([p["your_rating"] for p in recent])
        all_avg     = _avg([p["your_rating"] for p in self.papers])

        print(f"\n📉 Trend check")
        print(f"   All-time average   : {all_avg:.2f} / 5")
        print(f"   Recent 10 average  : {recent_avg:.2f} / 5")
        print(f"   Difference         : {recent_avg - all_avg:+.2f}")

        assert recent_avg >= all_avg - 0.5, (
            f"Recent average {recent_avg:.2f} is more than 0.5 below "
            f"all-time average {all_avg:.2f}. Ranking quality may be degrading."
        )
