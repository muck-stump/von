FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY . /app
WORKDIR /app

# uv.toml provides the IBM ppc64le index for torch; re-resolve without --frozen
# so uv picks the right wheels for this architecture
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "von", "serve", "--host", "0.0.0.0", "--port", "8000"]
