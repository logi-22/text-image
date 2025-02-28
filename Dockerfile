FROM python:3.9

# Set the working directory
WORKDIR /app

# Set an environment variable to redirect Hugging Face cache
ENV HF_HOME=/app/huggingface_cache

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
