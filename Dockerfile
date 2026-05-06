FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    pipx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user
RUN useradd -m -s /bin/bash nonroot && \
    usermod -aG sudo nonroot && \
    echo 'nonroot ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /workspace/ASOPA && \
    chown -R nonroot:nonroot /workspace/ASOPA

# Switch to non-root user
USER nonroot

# Install uv package manager for fast Python package management
RUN pipx install uv && \
    pipx ensurepath
ENV PATH="/home/nonroot/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace/ASOPA

# Copy lockfile + manifest first (better build-cache hit rate)
COPY pyproject.toml uv.lock* ./

# Install Python dependencies with GPU support via uv sync (uses [tool.uv.sources] cu121 index)
RUN uv sync --extra cu121 --frozen || uv sync --extra cu121

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose TensorBoard port for monitoring training
EXPOSE 6006
