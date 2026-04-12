FROM python:3.10-slim

WORKDIR /app

# Install system deps if needed
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

<<<<<<< HEAD
# Define a build-time argument (no secret in repo)
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Default command
CMD ["python", "inference.py"]
=======
# Default command
CMD ["python", "inference.py"]
>>>>>>> 080bf85532860b2c8ba89674fc41d041f4f9a5b2
