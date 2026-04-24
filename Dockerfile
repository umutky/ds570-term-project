# Base image: Use a slim Python version (better compatibility than alpine for ML libraries)
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (libgomp1 is required by LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the 'uv' package manager directly from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy dependency definition files and source code
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies exactly as locked in uv.lock (without dev dependencies)
RUN uv sync --frozen --no-dev

# Define the default command to run when the container starts
CMD ["uv", "run", "rf-train"]