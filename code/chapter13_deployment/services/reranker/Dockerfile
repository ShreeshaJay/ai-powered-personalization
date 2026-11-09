# Value function and re-ranker: Chapter 8. CPU only, no torch.
# Build from the chapter folder, naming the bundle to bake in:
#   docker build -f services/reranker/Dockerfile --build-arg BUNDLE=artifacts/<bundle_version> -t ch13-reranker .
# The value-function configs in config/reranker are baked in; in a container,
# promoting a config means a new image or a mounted RERANKER_CONFIG_DIR.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY services/reranker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY contracts/ contracts/
COPY common/ common/
COPY vendor/ vendor/
COPY config/reranker/ config/reranker/
COPY services/__init__.py services/
COPY services/reranker/ services/reranker/

ARG BUNDLE
COPY ${BUNDLE}/bundle_manifest.json bundle/
COPY ${BUNDLE}/reranker/ bundle/reranker/

RUN useradd --create-home --uid 10001 app
USER app
ENV CH13_BUNDLE_DIR=/app/bundle PORT=8080 OMP_NUM_THREADS=1
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn services.reranker.app:app --host 0.0.0.0 --port ${PORT}"]
