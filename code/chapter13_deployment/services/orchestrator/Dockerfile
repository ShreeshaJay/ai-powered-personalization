# Orchestrator: deadlines, budgets, circuit breakers and the degradation ladder. No torch.
# Build from the chapter folder, naming the bundle to bake in:
#   docker build -f services/orchestrator/Dockerfile --build-arg BUNDLE=artifacts/<bundle_version> -t ch13-orchestrator .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY services/orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY contracts/ contracts/
COPY common/ common/
COPY vendor/ vendor/
COPY services/__init__.py services/
COPY services/orchestrator/ services/orchestrator/

ARG BUNDLE
COPY ${BUNDLE}/bundle_manifest.json bundle/
COPY ${BUNDLE}/feature_store/ bundle/feature_store/
COPY ${BUNDLE}/orchestrator/ bundle/orchestrator/

RUN useradd --create-home --uid 10001 app
USER app
ENV CH13_BUNDLE_DIR=/app/bundle PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn services.orchestrator.app:app --host 0.0.0.0 --port ${PORT}"]
