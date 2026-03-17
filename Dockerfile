# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:25.08-py3

# System deps
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

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace/PhysMoDPO

# Copy lockfiles first for better layer caching
COPY pyproject.toml uv.lock ./

# Install all deps except torch/torchvision/torchaudio (already in NGC base)
RUN uv sync --frozen --no-install-project --no-group ngc-provided --system

# Copy repo code (includes OmniControl/ subdirectory)
COPY . .

# Download spaCy English model
RUN uv run --system python -m spacy download en_core_web_sm

ENV PYTHONPATH=/workspace/PhysMoDPO/OmniControl:/workspace/PhysMoDPO
ENV GRADIO_SERVER_PORT=4565
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 4565

CMD ["uv", "run", "--system", "python", "app.py"]
