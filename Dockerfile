# GPU Perf Prophet — CPU-only HF Spaces image (2 vCPU/16 GB free tier); build context is project root (data/models/, data/gpu_specs.yaml, data/pricing.yaml, src/, app/).

FROM python:3.11-slim

WORKDIR /app

# Install system deps required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements.txt changes); requirements.txt is production-only (dev-only pytest/pytest-cov/httpx live in requirements-dev.txt, never installed here).
COPY requirements.txt .
# xgboost's Linux wheel unconditionally pulls in nvidia-nccl-cu12 (~300 MB, CUDA-only, unneeded for this CPU-only tree_method="hist" image); pre-install xgboost --no-deps (version read from requirements.txt) so pip's second pass treats it as satisfied and skips resolving/downloading nvidia-nccl-cu12, with the trailing uninstall kept as an unverified-on-real-Linux safety net so the shipped image can't contain it either way.
RUN pip install --no-cache-dir --no-deps "$(grep '^xgboost==' requirements.txt)" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y nvidia-nccl-cu12

# Create a non-root user; HF Spaces also runs as UID 1000 by convention.
RUN useradd -m -u 1000 appuser

# Copy source and data artifacts
COPY src/       src/
COPY app/       app/
COPY data/gpu_specs.yaml  data/gpu_specs.yaml
COPY data/pricing.yaml    data/pricing.yaml
COPY data/models/         data/models/
COPY docker/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Transfer ownership before dropping privileges.
RUN chown -R appuser /app

USER appuser

# HF Spaces expects the app to listen on port 7860
ENV PORT=7860

# Streamlit (7860, HF Spaces' externally-routed port) and FastAPI (8000, internal — Streamlit's only client, see app/api_client.py) both run in this container; entrypoint.sh starts uvicorn in the background and execs streamlit in the foreground, so every UI interaction is real API traffic rather than an in-process call.
EXPOSE 7860
EXPOSE 8000

CMD ["./entrypoint.sh"]
