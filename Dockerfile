# Development environment with Jupyter Lab
FROM python:3.10-slim AS development

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create development user
RUN useradd -m -s /bin/bash -u 1000 jupyter_user && \
    chown -R jupyter_user:jupyter_user /home/jupyter_user

# Set working directory
WORKDIR /workspace

# Copy requirements files
COPY requirements*.txt ./

# Install Jupyter Lab and development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    jupyter-server-proxy \
    jupyterlab-git \
    black

# Switch to development user
USER jupyter_user

# Set up Jupyter Lab configuration
RUN mkdir -p /home/jupyter_user/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/jupyter_user/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /home/jupyter_user/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /home/jupyter_user/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /home/jupyter_user/.jupyter/jupyter_lab_config.py

# Add workspace bin to PATH
ENV PATH="/workspace/.local/bin:$PATH"

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]