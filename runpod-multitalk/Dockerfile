FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    xformers \
    flash-attn \
    librosa \
    transformers \
    diffusers \
    accelerate \
    omegaconf \
    einops \
    scipy \
    moviepy \
    opencv-python-headless \
    huggingface-hub

# Copy application code
COPY src/ /app/src/
COPY scripts/entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables for RunPod
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV MODEL_PATH=/runpod-volume/models

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]