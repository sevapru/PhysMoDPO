# syntax=docker/dockerfile:1
# Build context: repo root (set in docker-compose: context: .)
# Reason: the uv.lock lives at the workspace root, not inside this sub-repo.
FROM nvcr.io/nvidia/pytorch:25.08-py3

# System deps:
#   ffmpeg         — MP4 generation
#   libgl*         — headless OpenGL for pyrender/trimesh/open3d
#   libglib2.0-0   — GLib (OpenCV dep)
#   git, curl      — source package installs (CLIP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (static binary from official image, ARM64-native)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ── Workspace setup ──────────────────────────────────────────────────────────
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY PhysMoDPO/pyproject.toml /workspace/PhysMoDPO/

# ── Dependency install ────────────────────────────────────────────────────────
WORKDIR /workspace/PhysMoDPO
RUN uv sync --frozen --no-install-project --no-group ngc-provided --system

# ── Application code ─────────────────────────────────────────────────────────
COPY PhysMoDPO/ /workspace/PhysMoDPO/

# Download spaCy English model
RUN uv run --system python -m spacy download en_core_web_sm

# ── Runtime env ───────────────────────────────────────────────────────────────
# OmniControl subdir must be first on PYTHONPATH — it contains the importable
# modules (sample/, model/, diffusion/, data_loaders/, utils/)
ENV PYTHONPATH=/workspace/PhysMoDPO/OmniControl:/workspace/PhysMoDPO
ENV GRADIO_SERVER_PORT=4565
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 4565

CMD ["uv", "run", "--system", "python", "app.py"]
