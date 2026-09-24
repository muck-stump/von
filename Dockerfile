FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY . /app
WORKDIR /app

# uv.toml provides the IBM ppc64le index for torch; delete uv.lock so uv resolves
# fresh for this architecture and uses the IBM wheel index for torch
RUN rm -f uv.lock && uv sync --no-dev --default-index https://wheels.developerfirst.ibm.com/ppc64le/linux --index https://pypi.org/simple --index-strategy unsafe-best-match

EXPOSE 8000

CMD ["uv", "run", "von", "serve", "--host", "0.0.0.0", "--port", "8000"]
