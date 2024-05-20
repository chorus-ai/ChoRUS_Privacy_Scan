# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory inside the container
WORKDIR /privacy_scan_tool

# Install system dependencies required for h5py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements_batch.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements_batch.txt

# Copy the rest of the application code into the container
COPY . /privacy_scan_tool

# Run the application
CMD ["python", "main_batch.py"]

