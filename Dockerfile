FROM --platform=linux/amd64 python:3.11-slim
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY uv.lock .

# Install dependencies from pyproject.toml
# Sync core dependencies, but explicitly skip the heavy ML libraries
RUN uv sync --frozen --no-dev --no-editable
COPY . .

RUN mkdir -p logs
RUN mkdir -p /app/data && chmod 777 /app/data

CMD ["uv", "run", "python", "pipeline.py"]
