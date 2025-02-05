# Use a Python base image (we’ll add Node and Supervisor)
FROM python:3.9-slim

# Install Node.js (using Node 16 as an example) and Supervisor
RUN apt-get update && \
    apt-get install -y curl supervisor && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Set working directory
WORKDIR /app

# --- Set up the Python API ---
# Copy Python API code into container
COPY python-api/ ./python-api/

# Install Python dependencies
RUN pip install --no-cache-dir -r python-api/requirements.txt

# --- Set up the Next.js App ---
# Copy Next.js app code into container
COPY nextjs-frontend/ ./nextjs-frontend/

# Change directory to nextjs-frontend and install dependencies
WORKDIR /app/nextjs-frontend
RUN npm install

# (Optional) Build the Next.js app if you plan to run it in production mode.
RUN npm run build

# Return to the /app directory
WORKDIR /app

# --- Configure Supervisor ---
# Copy Supervisor configuration into the container
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports used by the applications
EXPOSE 5000 3000

# Start Supervisor (which will in turn start both services)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]