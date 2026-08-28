FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements-scraper.txt .
RUN pip install --no-cache-dir -r requirements-scraper.txt

# Copy application source code and utilities
COPY src/ ./src/
COPY utils/ ./utils/

# Copy processed dataset
COPY data/processed/posts_clean.parquet ./data/processed/posts_clean.parquet

# Set default entrypoint
ENTRYPOINT ["python", "-m", "src.video.main", "--bucket", "sm-optimizer-processed", "--platforms", "facebook", "instagram"]
