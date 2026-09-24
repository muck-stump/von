FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl gcc python3-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY . /app
WORKDIR /app

# uv.toml configures index-strategy = "unsafe-best-match" and the IBM ppc64le wheel index;
# delete uv.lock so uv re-resolves cleanly for this architecture
RUN rm -f uv.lock && uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "von", "serve", "--host", "0.0.0.0", "--port", "8000"]
