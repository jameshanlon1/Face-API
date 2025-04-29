# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /

# Update package lists and install required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Upgrade pip and install essential build tools
RUN pip install --upgrade pip setuptools wheel

# Install the Python dependencies
RUN pip install numpy pandas tqdm gdown opencv-python-headless
RUN pip install flask deepface
RUN pip install tf-keras
RUN pip install tflite-runtime
RUN pip install paho-mqtt
RUN pip install flask-cors


# Copy the rest of your app into the container
COPY . .

# Expose the port (change if needed)
EXPOSE 5000

# Command to run your app
CMD ["python", "verify-server.py"]
