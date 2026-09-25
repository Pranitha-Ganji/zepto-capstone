# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOCK_LLM=1 \
    PORT=7860

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY support_assistant/requirements.txt ./requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy module source code and corpus
COPY support_assistant/ /app/support_assistant/

# Run corpus ingestion to pre-build the ChromaDB vector index inside container image
RUN python -m support_assistant.ingest

# Expose port for local serving and Hugging Face Spaces compatibility
EXPOSE 7860

# Launch FastAPI service with uvicorn
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
