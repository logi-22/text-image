# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables to fix permission issues
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first (for caching dependencies)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Ensure the cache directory exists and has proper permissions
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
