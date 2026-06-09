FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Cloud Run uses PORT env variable
ENV PORT=8080

# Start server
CMD ["python", "main.py"]
