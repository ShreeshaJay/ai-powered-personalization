# Retrieval: Chapter 5 two-tower user tower + in-process FAISS index. CPU only.
# Build from the chapter folder, naming the bundle to bake in:
#   docker build -f services/retrieval/Dockerfile --build-arg BUNDLE=artifacts/<bundle_version> -t ch13-retrieval .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY services/retrieval/requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && pip install -r /tmp/requirements.txt

COPY contracts/ contracts/
COPY common/ common/
COPY vendor/ vendor/
COPY services/__init__.py services/
COPY services/retrieval/ services/retrieval/

ARG BUNDLE
COPY ${BUNDLE}/bundle_manifest.json bundle/
COPY ${BUNDLE}/retrieval/ bundle/retrieval/
COPY ${BUNDLE}/feature_store/ bundle/feature_store/

RUN useradd --create-home --uid 10001 app
USER app
ENV CH13_BUNDLE_DIR=/app/bundle PORT=8080 \
    OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1 FAISS_NUM_THREADS=1 TQDM_DISABLE=1
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn services.retrieval.app:app --host 0.0.0.0 --port ${PORT}"]
