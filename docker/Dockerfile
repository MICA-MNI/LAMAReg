FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic utilities
    wget \
    curl \
    unzip \
    git \
    build-essential \
    # ANTsPy dependencies
    cmake \
    # Neuroimaging dependencies
    libfreetype6-dev \
    pkg-config \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install LAMAReg in development mode
RUN pip install -e .

# Create a non-root user
RUN useradd -m -u 1000 lamareg && \
    chown -R lamareg:lamareg /app
USER lamareg

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/home/lamareg/.local/bin:$PATH

# Default command
CMD ["lamareg", "--help"]