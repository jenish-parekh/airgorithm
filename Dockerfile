# Use a lightweight base image with Python 3.10
FROM python:3.10-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy necessary directories explicitly
COPY air_api /air_api
COPY air_app /air_app
COPY air_package /air_package
COPY models /models
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

# Install project dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# Use a process manager to ensure both services run
CMD uvicorn air_api.api:app --host 0.0.0.0 --port $PORT
