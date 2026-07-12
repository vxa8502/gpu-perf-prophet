# GPU Perf Prophet — CPU-only HF Spaces compatible image
# Target: 2 vCPU / 16 GB RAM (HF Spaces free tier)
# Build context: project root (includes data/models/, data/gpu_specs.yaml,
#                data/pricing.yaml, src/, app/)

FROM python:3.11-slim

WORKDIR /app

# Install system deps required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements.txt changes).
# requirements.txt is production-only (see requirements-dev.txt for
# pytest/pytest-cov/httpx, deliberately never installed here).
COPY requirements.txt .
# xgboost's Linux wheel unconditionally declares
# `nvidia-nccl-cu12; platform_system == "Linux"` — a ~300 MB CUDA
# multi-GPU communication library — even though this image is CPU-only
# (tree_method="hist" only, no distributed/multi-GPU training or serving;
# this deployment requires no CUDA/ROCm dependency at runtime). Confirmed CPU
# train/predict/save/load all work identically without it.
#
# Pre-install xgboost with --no-deps first so pip (usually) never resolves
# — and therefore never downloads — nvidia-nccl-cu12 in the first place.
# The previous version of this fix installed everything normally then
# `pip uninstall`'d it, which fixed the shipped image size but still paid
# the full ~300 MB download + install + uninstall cost on every single
# build (measured ~5s+ of network transfer alone at 65 MB/s in that build
# log, thrown away every time). Safe because xgboost's other two real deps
# (numpy, scipy) are already pinned in requirements.txt below and get
# installed there — once xgboost is present at the exact pinned version,
# pip's second pass normally treats it as satisfied and doesn't re-resolve
# its dependency list. Version is read out of requirements.txt (not
# hardcoded here) so a future version bump can't silently desync the two.
#
# The final `pip uninstall` is kept as a safety net, not dropped: this
# specific pip resolver behavior could not be re-verified against a real
# Linux target in the session that made this change (Docker Desktop was
# unavailable in that sandbox) — an earlier attempt to verify it locally
# was itself invalid, since `platform_system == "Linux"` never evaluates
# true when pip runs on macOS, so it never actually exercised this path.
# If the pre-install trick fails to prevent the download for any reason,
# this line still guarantees the shipped image doesn't ship nvidia-nccl-cu12
# regardless — at worst, the build-time win doesn't materialize; the image
# stays correct either way. Re-verify the build-time claim in a real build.
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

# Transfer ownership before dropping privileges.
RUN chown -R appuser /app

USER appuser

# HF Spaces expects the app to listen on port 7860
ENV PORT=7860

# Expose both ports: Streamlit (7860) and FastAPI (8000)
EXPOSE 7860
EXPOSE 8000

# Default: run Streamlit UI
# Override CMD to run FastAPI instead:
#   docker run ... uvicorn src.api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
