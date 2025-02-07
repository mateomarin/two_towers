# Use a Python base image (we'll add Node and Supervisor)
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl supervisor \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

WORKDIR /app

# Copy and install Python API requirements first
COPY python-api/requirements.txt ./pythonapi/
RUN pip install --no-cache-dir -r pythonapi/requirements.txt

# Copy Python API files
COPY python-api/ ./pythonapi/
RUN mkdir -p /app/pythonapi/model
RUN touch /app/pythonapi/__init__.py
RUN touch /app/pythonapi/model/__init__.py
COPY python-api/utils.py ./pythonapi/
COPY python-api/model/margin_two_tower.py ./pythonapi/model/
COPY python-api/model/best_model.pt ./pythonapi/model/

# Copy Next.js frontend files
RUN mkdir -p /app/nextjs-frontend/pages /app/nextjs-frontend/styles
COPY nextjs-frontend/ ./nextjs-frontend/

# Install Next.js dependencies and build
WORKDIR /app/nextjs-frontend
RUN npm install
RUN npm run build

# Setup supervisor to run both services
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

# Expose ports for Python API (8000) and Next.js (3000)
EXPOSE 8000 3000

CMD ["/usr/bin/supervisord"]