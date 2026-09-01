.PHONY: install run test docker clean logs eval dashboard help

help:
	@echo ""
	@echo "  ScholarPulse — available commands"
	@echo ""
	@echo "  make install    Install dependencies with uv"
	@echo "  make run        Run the full pipeline (fetch → rank → email)"
	@echo "  make test       Run all 67 tests"
	@echo "  make eval       Evaluate ranker against your ground truth"
	@echo "  make logs       Show monitoring stats from all runs"
	@echo "  make dashboard  Open web dashboard at http://localhost:5000"
	@echo "  make docker     Build and run with Docker"
	@echo "  make clean      Remove cache files"
	@echo ""

install:
	uv sync

run:
	uv run python pipeline.py

test:
	uv run pytest tests/ -v -k "not judge"
 
judge:
	uv run pytest tests/test_llm_judge.py -v
eval:
	uv run python -c "\
import sys; sys.path.insert(0, '.'); \
from src.rag_ranker import RAGRanker; \
r = RAGRanker(); \
r.build_interest_profile(); \
r.evaluate(top_k=5)"

logs:
	uv run python -c "\
import sys; sys.path.insert(0, '.'); \
from src.monitor import Monitor; \
Monitor().print_stats()"

dashboard:
	uv run python dashboard.py

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "Cleaned."
