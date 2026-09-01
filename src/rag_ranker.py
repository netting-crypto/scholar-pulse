"""
rag_ranker.py

Agent 2: RAG Ranker
-------------------
Production-ready hybrid ranking system with optimized two-stage filtering.
"""

import os
import csv
import json
import requests
import numpy as np

from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Hardcoded lightweight metadata layer to fulfill project scope guidance
TOP_RESEARCHERS = [
    "Fei-Fei Li", "Yann LeCun", "Andrew Ng", "Demis Hassabis", 
    "Kaiming He", "Ross Girshick", "Chelsea Finn", "Sergey Levine"
]

# ============================================================
# LLM CLIENT
# ============================================================

def call_llm(prompt: str) -> Dict:
    """Compatible with OpenAI, Groq, and Ollama endpoints."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    api_base = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE") or os.getenv("GROQ_API_BASE")
    model = os.getenv("LLM_API_MODEL") or os.getenv("OPENAI_API_MODEL") or "gpt-3.5-turbo"

    if os.getenv("GROQ_API_KEY") and not api_base:
        api_base = "https://api.groq.com/openai/v1"
    if not api_base:
        api_base = "https://api.openai.com/v1"

    if not api_key:
        return {
            "score": 5, "reason": "LLM disabled", "rationale": "", "highlights": [],
            "error_code": "NO_API_KEY", "detail": "Missing API key"
        }

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a research-paper relevance evaluator. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 700,
        "response_format": {"type": "json_object"}  # Hard-enforce JSON return structure
    }

    try:
        response = requests.post(
            url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=15
        )
        if response.status_code != 200:
            return {"score": 5, "reason": "LLM HTTP error", "error_code": "HTTP_ERROR", "detail": response.text[:500]}

        data = response.json()
        text = data["choices"][0]["message"]["content"].replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"score": 5, "reason": "LLM Exception", "error_code": "REQUEST_EXCEPTION", "detail": str(e)}


# ============================================================
# RAG RANKER
# ============================================================

class RAGRanker:

    def __init__(
        self,
        ground_truth_path="data/ground_truth_papers.csv",
        use_llm=True,
        min_rating=3,
        model_name="paraphrase-MiniLM-L3-v2",
    ):
        self.ground_truth_path = ground_truth_path
        self.use_llm = use_llm
        self.min_rating = min_rating
        self.model = SentenceTransformer(model_name)
        self.ground_truth = []
        self.profile_embedding = None
        self.interest_summary = ""

    def _load_ground_truth(self):
        encodings = ["utf-8", "cp1252", "latin1"]
        for enc in encodings:
            try:
                with open(self.ground_truth_path, "r", encoding=enc) as f:
                    self.ground_truth = list(csv.DictReader(f))
                break
            except Exception:
                continue

        if not self.ground_truth:
            raise RuntimeError(f"Unable to load CSV: {self.ground_truth_path}")

        for paper in self.ground_truth:
            try:
                paper["your_rating"] = float(paper.get("your_rating", 0))
            except Exception:
                paper["your_rating"] = 0.0

    def build_interest_profile(self):
        self._load_ground_truth()
        positive = [p for p in self.ground_truth if p["your_rating"] >= self.min_rating]
        if not positive:
            raise ValueError("No positively rated papers found.")

        texts = [f"{p['title']} {p['abstract']}" for p in positive]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        profile = np.mean(embeddings, axis=0)
        self.profile_embedding = profile / (np.linalg.norm(profile) + 1e-8)

        topics = set()
        for p in positive:
            topics.update(p.get("why_you_care", "").split(","))
        topics = [t.strip() for t in topics if t.strip()]
        self.interest_summary = "User interests: " + ", ".join(topics)
        print(f"✅ Profile built successfully from {len(positive)} papers.")

    def _embed_papers(self, papers: List[Dict]):
        texts = [f"{p.get('title','')} {p.get('abstract','')}" for p in papers]
        return self.model.encode(texts, normalize_embeddings=True)

    def _apply_metadata_multiplier(self, paper: Dict, score: float) -> float:
        """Applies a lightweight metadata boost for top industry researchers."""
        authors_list = paper.get("authors", [])
        # Handle string variations or actual structural lists
        if isinstance(authors_list, str):
            authors_list = [authors_list]
            
        for author in authors_list:
            if any(top_author.lower() in author.lower() for top_author in TOP_RESEARCHERS):
                return min(score * 1.15, 1.0)  # Max out multiplier at ceiling 1.0
        return score

    def rank_papers(self, papers: List[Dict], top_k=5):
        if self.profile_embedding is None:
            raise ValueError("Call build_interest_profile() first")

        if not papers:
            return []

        print(f"\n🔍 Stage 1: Semantic Filter tracking {len(papers)} papers...")
        embeddings = self._embed_papers(papers)
        
        stage1_scored = []
        for paper, emb in zip(papers, embeddings):
            semantic_score = float(np.dot(self.profile_embedding, emb))
            semantic_score = (semantic_score + 1.0) / 2.0  # Normalize to [0,1]
            semantic_score = min(max(semantic_score, 0.0), 1.0)
            
            # Injecting lightweight feature multiplier
            semantic_score = self._apply_metadata_multiplier(paper, semantic_score)
            
            paper_copy = paper.copy()
            paper_copy["semantic_score"] = round(semantic_score, 4)
            stage1_scored.append(paper_copy)

        # Sort based on Stage 1 Semantic metrics
        stage1_scored.sort(key=lambda x: x["semantic_score"], reverse=True)
        
        # CRITICAL FIX: Only pass candidates clearing Stage 1 to Stage 2 LLM
        # We process the top (top_k * 2) or maximum 10 papers via the LLM to cut costs and time
        candidates_to_rerank = stage1_scored[:max(top_k * 2, 10)]
        print(f"🧠 Stage 2: Reranking top {len(candidates_to_rerank)} items via LLM Context Judgement...")

        final_ranked = []
        for paper in candidates_to_rerank:
            llm_score = 0.0
            reason, rationale, highlights = "LLM disabled", "", []

            if self.use_llm:
                prompt = f"""Evaluate research paper relevance.\n\nUser interests:\n{self.interest_summary}\n\nPaper:\nTitle: {paper.get('title', '')}\nAbstract:\n{paper.get('abstract', '')[:1200]}\n\nReturn ONLY valid JSON:\n{{\n  "score": "integer between 1 and 10",\n  "reason": "one sentence summary",\n  "rationale": "3-5 sentence explanation",\n  "highlights": ["keyword1", "keyword2"]\n}}"""
                result = call_llm(prompt)
                
                if "error_code" not in result:
                    try:
                        llm_score = min(max(float(result.get("score", 5)) / 10.0, 0.0), 1.0)
                        reason = result.get("reason", "")
                        rationale = result.get("rationale", "")
                        highlights = result.get("highlights", [])
                    except Exception:
                        llm_score = 0.5

            final_score = 100 * (0.6 * paper["semantic_score"] + 0.4 * llm_score)
            
            paper["llm_score"] = round(llm_score, 4)
            paper["final_score"] = round(final_score, 2)
            paper["llm_reason"] = reason
            paper["llm_rationale"] = rationale
            paper["llm_highlights"] = highlights if isinstance(highlights, list) else [str(highlights)]
            final_ranked.append(paper)

        final_ranked.sort(key=lambda x: x["final_score"], reverse=True)
        top_candidates = final_ranked[:top_k]

        print(f"\n🏆 Top {len(top_candidates)} Papers Identified:")
        for i, paper in enumerate(top_candidates, start=1):
            print(f"  {i}. [{paper['final_score']}] {paper['title'][:70]}...")

        return top_candidates

    def _ndcg_at_k(self, ranked, ratings, k):
        dcg = 0.0
        # Normalize keys for fallback accuracy check
        norm_ratings = {k.strip().lower(): v for k, v in ratings.items()}
        
        for i, paper in enumerate(ranked[:k]):
            title_key = paper["title"].strip().lower()
            rel = norm_ratings.get(title_key, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)

        ideal_rels = sorted(ratings.values(), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, top_k=5):
        print("\n📊 Beginning Offline Evaluation Routine...")
        self._load_ground_truth()

        papers = [{"title": p["title"], "abstract": p["abstract"], "authors": p.get("authors", [])} for p in self.ground_truth]
        ratings = {p["title"]: p["your_rating"] for p in self.ground_truth}
        
        # Safe string normalization layer
        relevant_titles = {p["title"].strip().lower() for p in self.ground_truth if p["your_rating"] >= 4}

        ranked = self.rank_papers(papers, top_k=top_k)
        hits = sum(1 for p in ranked if p["title"].strip().lower() in relevant_titles)

        precision = hits / top_k
        recall = hits / len(relevant_titles) if relevant_titles else 0.0
        ndcg = self._ndcg_at_k(ranked, ratings, top_k)

        print(f"\n📈 Results -> Precision@{top_k}: {precision:.3f} | Recall@{top_k}: {recall:.3f} | NDCG@{top_k}: {ndcg:.3f}")
        return {"precision": round(precision, 4), "recall": round(recall, 4), "ndcg": round(ndcg, 4), "hits": hits, "top_k": top_k}