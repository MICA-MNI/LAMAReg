FROM python:3.11-slim

# Build arguments for server environment
ARG CUSTOM_TMPDIR=/tmp
ARG BUILDKIT_INLINE_CACHE=1

# Set temporary directory for server builds
ENV TMPDIR=${CUSTOM_TMPDIR}
ENV TMP=${CUSTOM_TMPDIR}
ENV TEMP=${CUSTOM_TMPDIR}

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
    # Additional server dependencies
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p ${CUSTOM_TMPDIR}

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