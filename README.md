# ScholarPulse — Personal AI Research Scout

Monitors ArXiv daily, ranks papers by **your** interests using RAG, and emails you a
personalized digest every morning. You rate papers directly in the email; ratings
feed back into the ranking model automatically.

---

## Problem

Researchers publishing in fast-moving fields like LLMs, agents, and multimodal AI
face a daily flood of ArXiv papers — often 100–300 new submissions per day in a
single category. Manually scanning titles and abstracts wastes hours and still
misses relevant work.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/cce0476b-0bd0-4955-ab3a-adfcd1b93b87" />

**ScholarPulse solves this by:**
1. Fetching every new paper from your chosen ArXiv categories each morning
2. Ranking them against a personalized interest profile built from papers *you*
   have rated highly in the past
3. Emailing you a digest of only the top matches — with relevance scores and
   explanations — so you read what matters and skip the rest
4. Learning from your in-email star ratings to improve rankings over time
<img width="950" height="410" alt="image" src="https://github.com/user-attachments/assets/44fb08f6-d063-4f98-8267-a6eaee368a0e" />

---

## Architecture

```
ArXiv API → Agent 1 (Fetch) → Agent 2 (RAG Rank) → Agent 3 (Email) + Monitor
                                        ↑
                              Feedback Server ← ⭐ Email ratings
                                        ↓
                           ground_truth_papers.csv (auto-updated)
```

| File | Role |
|---|---|
| `src/agent_arxiv.py` | Fetches today's papers from ArXiv |
| `src/rag_ranker.py` | Two-stage RAG ranker (semantic + LLM rerank) |
| `src/digest_sender.py` | Formats and sends HTML email via Resend |
| `src/feedback_server.py` | FastAPI webhook — receives star ratings from emails |
| `src/monitor.py` | Logs every run to `logs/runs.jsonl` |
| `pipeline.py` | Wires all agents together |
| `dashboard.py` | Live monitoring dashboard at `http://localhost:5000` |

---

## Agent Tools

| Agent | Tools Used |
|---|---|
| Agent 1 (Fetch) | ArXiv API, HTTP requests |
| Agent 2 (RAG Rank) | `sentence-transformers`, semantic embeddings, OpenAI-compatible LLM |
| Agent 3 (Email) | Resend API, HTML email formatting |
| Feedback Server | FastAPI webhook, CSV read/write, profile rebuilder |
| Dashboard | Live UI, JSONL run logs, monitoring charts |

---

## Knowledge Base & Retrieval

### Knowledge base

`data/ground_truth_papers.csv` is a **hand-curated** dataset of papers rated 1–5
by the user. It has these columns:

| Column | Description |
|---|---|
| `arxiv_id` | ArXiv paper ID |
| `title` | Paper title |
| `abstract` | Full abstract |
| `your_rating` | Manual rating 1–5 (5 = must read) |
| `why_you_care` | Keyword tags explaining relevance |

This file is used both as the **retrieval knowledge base** and as the
**evaluation ground truth**. It grows automatically as you rate papers from
your daily digests.

### Two-stage retrieval

**Stage 1 — Semantic search (RAG)**
- Encodes all positively-rated papers (rating ≥ 3) with
  `paraphrase-MiniLM-L3-v2` (sentence-transformers)
- Builds a mean interest profile embedding from those encodings
- Scores every incoming ArXiv paper by cosine similarity to the profile
- Applies a 1.15× boost for papers by top researchers
- Passes top `2 × top_k` candidates to Stage 2

**Stage 2 — LLM reranking**
- Sends each candidate to an OpenAI-compatible LLM with the user's interest
  summary and the paper abstract
- LLM returns a relevance score (1–10), a one-sentence reason, and keyword
  highlights
- Final score: `0.6 × semantic_score + 0.4 × llm_score` (scaled to 0–100)
<img width="390" height="583" alt="image" src="https://github.com/user-attachments/assets/c65ddfe8-edee-4760-a72c-4da26b599c97" />

### Retrieval evaluation

ScholarPulse evaluates ranking quality using held-out relevance from
`data/ground_truth_papers.csv`. This file acts as the ground truth for papers
you have rated, so the evaluation measures how well the model puts your
highly-rated papers near the top.

**Metrics explained**

| Metric | What it measures | Good value means |
|---|---|---|
| `Precision@K` | Of the top `K` recommended papers, what fraction are in your high-relevance ground truth set? | Higher is better; `1.0` means all top-K papers are relevant |
| `Recall@K` | Of all relevant ground truth papers, what fraction appear in the top `K` recommendations? | Higher is better; `1.0` means the model found every relevant paper |
| `NDCG@K` | Rank-sensitive quality score for the top `K` results, weighting higher positions more strongly | Higher is better; `1.0` is perfect ordering, values above `0.7` are typically strong for noisy relevance data |

This evaluation is especially useful because it directly compares system output
against the papers you have already rated rather than only relying on absolute
score numbers.

We used evaluation to tune key parameters:
- semantic / LLM score blend weights (`0.6 / 0.4`)
- Stage 1 candidate pool size (`top_k × 2`)
- minimum rating threshold for profile building (`min_rating=3`)

Run evaluation:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from src.rag_ranker import RAGRanker
r = RAGRanker()
r.build_interest_profile()
r.evaluate(top_k=5)
"
```

Example output on the current dataset:
```
📈 Results -> Precision@5: 0.800 | Recall@5: 0.667 | NDCG@5: 0.923
```

**Interpretation of this example**
- `Precision@5: 0.800` means 4 of the top 5 recommended papers matched your
  high-relevance ground truth papers.
- `Recall@5: 0.667` means the top 5 recommendations captured roughly two-thirds
  of the relevant papers from the ground truth set.
- `NDCG@5: 0.923` means the recommendations are not only relevant, but also
  well-ordered, with the most relevant papers ranked near the top.

> Note: actual evaluation scores change as you add more ratings to
> `data/ground_truth_papers.csv` and as the current ranking pipeline evolves.

---

## Quick Start (local)

### 1. Install dependencies

```bash
# Requires Python 3.10+ and uv
uv sync
```

### 2. Set environment variables

```powershell
# Windows PowerShell
$env:RESEND_API_KEY   = "re_xxxxxxxxxxxxxxxx"   # get free at resend.com
$env:DIGEST_TO        = "you@gmail.com"
$env:LLM_API_KEY      = "your-api-key"
$env:LLM_API_BASE     = "https://api.groq.com/openai/v1"  # optional, Groq is free
$env:FEEDBACK_BASE_URL = "http://localhost:8000"  # or your ngrok URL
```

```bash
# Linux / macOS
export RESEND_API_KEY="re_xxxxxxxxxxxxxxxx"
export DIGEST_TO="you@gmail.com"
export LLM_API_KEY="your-api-key"
export LLM_API_BASE="https://api.groq.com/openai/v1"
export FEEDBACK_BASE_URL="http://localhost:8000"
```

### 3. Run the pipeline

```bash
uv run python pipeline.py
```

### 4. Start the feedback server (to receive email ratings)

```bash
uv run uvicorn src.feedback_server:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the monitoring dashboard

```bash
uv run python dashboard.py
# → http://localhost:5000
```

---

## Quick Start (Docker)

```bash
# 1. Copy and fill in your keys
cp .env.example .env

# 2. Start everything
docker compose up
```

`docker compose up` starts:
- The pipeline runner (sends the daily digest)
- The feedback server on port 8000
- The monitoring dashboard on port 5000

---

## Cloud Deployment

This project is Docker-ready and can be deployed to cloud providers such as Railway, Render, or any Docker-compatible host. Use `docker compose up` locally, then deploy the same `Dockerfile` and `docker-compose.yml` to run the pipeline, feedback server, and dashboard together.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `RESEND_API_KEY` | Yes | Get free at resend.com |
| `DIGEST_TO` | Yes | Your email address |
| `LLM_API_KEY` | No | OpenAI-compatible API key |
| `LLM_API_BASE` | No | Alternate endpoint, e.g. `https://api.groq.com/openai/v1` |
| `LLM_API_MODEL` | No | Model name, default `gpt-3.5-turbo` |
| `FEEDBACK_BASE_URL` | No | Public URL of feedback server (default `http://localhost:8000`) |
| `FEEDBACK_CSV` | No | Path to ground truth CSV (default `data/ground_truth_papers.csv`) |

Free compatible providers: Groq works out of the box. Set
`LLM_API_BASE=https://api.groq.com/openai/v1` and use your Groq key in
`LLM_API_KEY`.

---

## Testing

### GitHub Actions CI

A GitHub Actions workflow runs `uv run pytest tests/ -v` on every push and pull request.

### Run all unit tests

```bash
uv run pytest tests/ -v
# Expected: 67 passed
```

### Run satisfaction evaluation tests

These tests measure system quality using **real user ratings** from your
digest emails — no API key required.

```bash
uv run pytest tests/test_llm_judge.py -v -s
```

The `-s` flag prints the full satisfaction report inline:

```
📊 User Satisfaction Score
   Total rated papers : 47
   Average rating     : 4.10 / 5.00

⭐ High-relevance ratio (rating 4-5)
   Papers rated 4-5 : 32 / 47    (68%)

📈 Rating distribution
   Must read     ⭐⭐⭐⭐⭐ :  18 ( 38.3%) ███████
   Interesting   ⭐⭐⭐⭐  :  14 ( 29.8%) █████
   Maybe         ⭐⭐⭐   :   9 ( 19.1%) ███
   Not relevant  ⭐⭐     :   4 (  8.5%) █
   Irrelevant    ⭐       :   2 (  4.3%)
```

The satisfaction tests check:

| Test | Threshold | Meaning |
|---|---|---|
| Average rating | ≥ 3.0 / 5 | System is not recommending mostly irrelevant papers |
| High-relevance ratio | ≥ 40% rated 4-5 | Enough must-read / interesting papers |
| Low-relevance ratio | < 30% rated 1-2 | Not too many irrelevant papers |
| Distribution report | always passes | Prints full breakdown for documentation |
| Recent trend | within 0.5 of all-time avg | Quality is not degrading over time |

The last test activates automatically once you have 10+ timestamped ratings.

---

## Evaluation

The ranker is evaluated against `data/ground_truth_papers.csv` — a set of
papers **manually rated 1–5** by the user (not LLM-generated).

**Metrics:**
- **Precision@K** — of the top K recommended papers, how many match your
  highly-rated ground truth papers (rating ≥ 4)?
- **Recall@K** — of all highly-rated papers, how many appear in the top K?
- **NDCG@K** — normalized discounted cumulative gain, accounts for rank order

**How we used evaluation to tune parameters:**

We ran `evaluate(top_k=5)` after each change and compared NDCG@5:

| Change tested | NDCG@5 before | NDCG@5 after |
|---|---|---|
| Blend weights 0.5/0.5 → 0.6/0.4 | 0.871 | 0.923 |
| Candidate pool `top_k` → `top_k × 2` | 0.884 | 0.923 |
| `min_rating` 4 → 3 | 0.901 | 0.923 |

Run evaluation:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from src.rag_ranker import RAGRanker
r = RAGRanker()
r.build_interest_profile()
results = r.evaluate(top_k=5)
print(results)
"
```

---

## User Feedback Loop

Every digest email contains **1★ – 5★★★★★** rating buttons per paper.
Clicking a button:

1. Sends a GET request to the feedback server (`/feedback`)
2. The server writes or updates the row in `ground_truth_papers.csv` with:
   - `your_rating` = the star rating you clicked (1–5)
   - `why_you_care` = the paper's keyword highlights from the LLM
3. The ranker's interest profile is rebuilt immediately

This means **every rating you click directly improves tomorrow's digest**
and simultaneously expands the ground truth dataset used for evaluation.

To view saved papers: `http://localhost:8000/saved`
To view rating stats: `http://localhost:8000/stats`

---

## UI Dashboard & Monitoring

ScholarPulse includes a live dashboard UI for monitoring pipeline runs, relevance trends, and saved ratings.

Every pipeline run is logged to `logs/runs.jsonl`. Each line is a JSON event:

```json
{"event": "recommendation", "run_id": "2026-06-11T08:00:00",
 "rank": 1, "title": "...", "final_score": 87.5, "timestamp": "..."}
```

### Access the dashboard

```bash
uv run python dashboard.py
# → http://localhost:5000
```
<img width="938" height="442" alt="image" src="https://github.com/user-attachments/assets/60fe878b-9a09-4942-9a78-479a55e80294" />
<img width="907" height="424" alt="image" src="https://github.com/user-attachments/assets/e9939ade-1fbc-4dde-9838-aebf759693be" />


The dashboard shows:
- Total runs, papers ranked, average relevance score, errors
- Relevance score trend over the last 14 runs (line chart)
- Recent recommendations with scores and links
- Full run history table

Switch to the **My ratings** tab to see:
- Papers rated (total, highly relevant, saved)
- Stacked bar chart of ratings over time
- Saved-for-later list
- Full rating history filterable by star level

### Logs → ground truth → evaluation pipeline

Ratings collected via the feedback server are written directly into
`ground_truth_papers.csv`. This means you can run `r.evaluate(top_k=5)`
at any time to measure how well the current ranker performs against all
papers you have ever rated — closing the loop from monitoring to evaluation
automatically.

---

## Code Structure

```
scholar_pulse/
├── src/
│   ├── agent_arxiv.py       # Agent 1: ArXiv fetcher
│   ├── rag_ranker.py        # Agent 2: Two-stage RAG ranker
│   ├── digest_sender.py     # Agent 3: Email digest sender
│   ├── feedback_server.py   # FastAPI feedback webhook
│   └── monitor.py           # Run logger
├── tests/
│   ├── test_arxiv.py        # Unit tests for ArXiv agent
│   ├── test_ranker.py       # Unit tests for RAG ranker
│   ├── test_digest.py       # Unit tests for digest sender
│   └── test_llm_judge.py    # LLM-as-judge evaluation tests
├── data/
│   └── ground_truth_papers.csv   # Hand-curated paper ratings
├── logs/
│   └── runs.jsonl           # Pipeline run logs
├── pipeline.py              # Main pipeline entry point
├── dashboard.py             # Monitoring dashboard
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── uv.lock
```

---

## Makefile

```bash
make run              # Run the pipeline
make serve-feedback   # Start feedback server
make dashboard        # Start monitoring dashboard
make test             # Run all unit tests
make judge            # Run LLM-as-judge tests
make evaluate         # Run Precision/Recall/NDCG evaluation
```

---

## Best Practices

- **Dependency management**: `uv` with `pyproject.toml` and `uv.lock`
- **Containerization**: `Dockerfile` + `docker-compose.yml`
- **Testing**: unit tests (67 passing) + LLM-as-judge tests
- **Evaluation**: hand-crafted ground truth + Precision@K / NDCG@K
- **Monitoring**: structured JSONL logs + live dashboard
- **Feedback loop**: email ratings → CSV → profile rebuild → better rankings
