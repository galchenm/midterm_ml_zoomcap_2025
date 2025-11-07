FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app
COPY requirements.txt .

# Use uv for dependency install
RUN uv pip install -r requirements.txt

#COPY . .
#CMD ["python", "scripts/app.py"]
