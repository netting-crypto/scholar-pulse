"""
ScholarPulse — Full Pipeline
Agent 1 (ArXiv) → Agent 2 (Ranker) → Agent 3 (Email) + Monitoring

Run:
    python pipeline.py

With Docker:
    docker compose up
"""

import os
import sys

# Forces Python to check the directory of this file for module lookups
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip configuration injection

from datetime import datetime
from src.agent_arxiv  import ArXivMonitor
# FIXED: Updated module import paths to target rewritten files
from src.rag_ranker   import RAGRanker
from src.digest_sender import DigestSender
from src.monitor      import Monitor

MY_KEYWORDS = [
    "vision language model", "multimodal", "VLM",
    "AI agent", "agentic", "healthcare", "medical imaging",
    "document understanding", "embodied AI", "robotic",
]
MY_CATEGORIES = ["cs.CV", "cs.AI", "cs.LG", "cs.CL", "cs.RO"]
FETCH_COUNT   = 50
TOP_K         = 5


def run():
    use_llm  = bool(
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    has_mail = bool(os.environ.get("RESEND_API_KEY"))
    api_base = (
        os.getenv("LLM_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("GROQ_API_BASE")
        or "https://api.openai.com/v1"
    )
    monitor  = Monitor()

    print("=" * 62)
    print("   ScholarPulse — Daily Digest")
    print(f"   LLM  : {'ON' if use_llm  else 'OFF (set LLM_API_KEY or OPENAI_API_KEY)'}")
    print(f"   API  : {api_base}")
    print(f"   Email: {'ON' if has_mail else 'OFF (set RESEND_API_KEY)'}")
    print("=" * 62)

    try:
        # ── Agent 1: Fetch ──────────────────────────────────────────────────
        monitor_arxiv = ArXivMonitor(keywords=MY_KEYWORDS, categories=MY_CATEGORIES)
        papers = monitor_arxiv.fetch(max_results=FETCH_COUNT)
        if not papers:
            monitor.log_error("No papers fetched")
            print("No papers fetched. Check your internet connection.")
            return

        monitor.log_run_start(papers_fetched=len(papers))

        # ── Agent 2: Rank ───────────────────────────────────────────────────
        ranker = RAGRanker(
            ground_truth_path="data/ground_truth_papers.csv",
            use_llm=use_llm,
        )

        ranker.build_interest_profile()

        top_papers = ranker.rank_papers(
            papers,
            top_k=TOP_K
        )

        # Optional evaluation on your ground truth parameters matrix
        try:
            eval_results = ranker.evaluate(top_k=TOP_K)

            print("\nEvaluation Metrics")
            print("------------------")
            print(f"Precision@{TOP_K}: {eval_results['precision']:.3f}")
            print(f"Recall@{TOP_K}:    {eval_results['recall']:.3f}")
            print(f"NDCG@{TOP_K}:      {eval_results['ndcg']:.3f}")

        except Exception as e:
            print(f"Evaluation skipped: {e}")

        # Log recommendations to your jsonl history tracking block
        monitor.log_recommendations(top_papers)

        stats = {
            "date"         : datetime.now().strftime("%B %d, %Y"),
            "total_fetched": len(papers),
            "match_rate"   : f"{TOP_K / len(papers):.1%}",
        }

        # ── Terminal Output Digest ──────────────────────────────────────────
        print("\n" + "=" * 62)
        print(f"   TOP {TOP_K} PAPERS FOR YOU")
        print("=" * 62)
        for i, p in enumerate(top_papers, 1):
            print(f"\n  {i}. {p['title']}")
            print(f"     Score     : {p['final_score']}/100")
            print(f"     Semantic  : {p['semantic_score']}")
            print(f"     LLM Score : {p.get('llm_score', 'N/A')}")
            # FIXED: Pointing output to matched llm_reason key tracking variable
            print(f"     Reason    : {p.get('llm_reason', 'N/A')}")
            
            print("     Highlights:")
            highlights = p.get("llm_highlights", [])
            if isinstance(highlights, list) and highlights:
                for h in highlights:
                    print(f"        - {h}")
            else:
                print("        - None")

            print(f"     URL       : {p.get('url', 'N/A')}")

        # ── Agent 3: Send email ─────────────────────────────────────────────
        print("\n" + "=" * 62)
        sender     = DigestSender()
        email_sent = sender.send(top_papers, stats)

        # Log run end metrics data profiles
        monitor.log_run_end(success=True, email_sent=email_sent)
        monitor.print_stats()

    except Exception as e:
        monitor.log_error(str(e))
        monitor.log_run_end(success=False)
        raise


if __name__ == "__main__":
    run()
