# Use a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python package
COPY . .

# Install the PipeLM package
RUN pip install -e .

# Create a volume for model storage
VOLUME /root/.pipelm

# Set environment variables
ENV HF_HOME=/root/.pipelm/cache
ENV TRANSFORMERS_CACHE=/root/.pipelm/cache/transformers
ENV MODEL_DIR=""
ENV PORT=8080

# Default command to run the server
ENTRYPOINT ["python", "-m", "pipelm"]
CMD ["server", "--port", "8080"]

# Expose the API port
EXPOSE 8080