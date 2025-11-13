FROM python:3.11-slim

# Install system dependencies required by numpy/pandas/xgboost
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app
COPY requirements.txt .

# Use uv for dependency install
RUN uv pip install --system -r requirements.txt

# Copy the application code and model artifacts
COPY app ./app
COPY model ./model

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
