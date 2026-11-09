# Lightweight ranker: Chapter 6 XGBoost. CPU only, no torch.
# Build from the chapter folder, naming the bundle to bake in:
#   docker build -f services/lightweight_ranker/Dockerfile --build-arg BUNDLE=artifacts/<bundle_version> -t ch13-light-ranker .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY services/lightweight_ranker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY contracts/ contracts/
COPY common/ common/
COPY vendor/ vendor/
COPY services/__init__.py services/
COPY services/lightweight_ranker/ services/lightweight_ranker/

ARG BUNDLE
COPY ${BUNDLE}/bundle_manifest.json bundle/
COPY ${BUNDLE}/lightweight_ranker/ bundle/lightweight_ranker/
COPY ${BUNDLE}/feature_store/feature_contract.json bundle/feature_store/

RUN useradd --create-home --uid 10001 app
USER app
ENV CH13_BUNDLE_DIR=/app/bundle PORT=8080 OMP_NUM_THREADS=1 XGB_NUM_THREADS=1
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn services.lightweight_ranker.app:app --host 0.0.0.0 --port ${PORT}"]
