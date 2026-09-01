"""
Quick demo:

    python demo_ranker.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.rag_ranker import RAGRanker


NEW_PAPERS = [
    {
        "arxiv_id": "2501.11111",
        "title": "Vision-Language Models for Radiology Report Generation",
        "abstract": (
            "We propose a multimodal transformer that generates "
            "structured radiology reports from chest X-rays."
        ),
    },
    {
        "arxiv_id": "2501.22222",
        "title": "Agentic LLMs with Tool Use for Healthcare Automation",
        "abstract": (
            "Language models autonomously call medical APIs "
            "and generate clinical summaries."
        ),
    },
    {
        "arxiv_id": "2501.33333",
        "title": "Efficient Sorting Networks on Modern GPUs",
        "abstract": (
            "We analyze parallel GPU sorting methods."
        ),
    },
    {
        "arxiv_id": "2501.44444",
        "title": "Robotic Arm Control via Vision-Language Instructions",
        "abstract": (
            "Robots follow natural language commands."
        ),
    },
]


def main():

    use_llm = True
    

    print("=" * 60)
    print("🔭 SCHOLARPULSE — RAG Ranker Demo")
    print(
        f"LLM scoring: "
        f"{'✅ ON (Llama 3)' if use_llm else '⚠️ OFF'}"
    )
    print("=" * 60)

    ranker = RAGRanker(
        ground_truth_path="ground_truth_papers.csv",
        use_llm=use_llm,
    )

    ranker.build_interest_profile()

    top_papers = ranker.rank_papers(
        NEW_PAPERS,
        top_k=4,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("📋 TODAY'S DIGEST")
    print("=" * 60)

    for i, paper in enumerate(top_papers, 1):

        print(f"\n{i}. {paper['title']}")
        print(f"Score : {paper['final_score']}/100")
        print(
            f"ArXiv : "
            f"https://arxiv.org/abs/{paper['arxiv_id']}"
        )
        print(f"Why   : {paper['explanation']}")

    print("\n" + "=" * 60)
    print("📊 SELF-EVALUATION")
    print("=" * 60)

    metrics = ranker.evaluate(top_k=5)

    print(
        f"\n✅ Precision@5 : "
        f"{metrics['precision_at_k']:.0%}"
    )

    print(
        f"✅ Score gap   : "
        f"{metrics['score_gap']:.1f}"
    )

    target = 0.6

    status = (
        "✅ PASS"
        if metrics["precision_at_k"] >= target
        else "⚠️ Below target"
    )

    print(f"\nResult: {status}")


if __name__ == "__main__":
    main()
