# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    ffmpeg \
    libgl1-mesa-dev \
    libxrender1 \
    libsm6 \
    libxext6 \
    libxfixes3 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy application files
COPY gui/ /app/gui
COPY tools/ /app/tools
COPY requirements.txt /app/
COPY entrypoint.sh /app/

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create a non-root user for better security
RUN useradd -m appuser && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Define the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
