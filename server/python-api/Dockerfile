FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and code
COPY app.py .
COPY margin_two_tower.py .
COPY utils.py .
COPY dataset_ms_marco.py .
COPY best_model.pt .

# Create cache directory
RUN mkdir -p /app/cache

# Run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 