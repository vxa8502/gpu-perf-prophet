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

# Install Python deps first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
