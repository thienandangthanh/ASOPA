FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

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

# Copy project files
COPY requirements.txt .

# Create virtual environment with uv
RUN uv venv .venv

# Install Python dependencies with GPU support
RUN . .venv/bin/activate && \
    uv pip install -r requirements.txt \
    --index https://download.pytorch.org/whl/cu129

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose TensorBoard port for monitoring training
EXPOSE 6006
