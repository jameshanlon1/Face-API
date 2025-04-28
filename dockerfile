# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /

# Update package lists and install required system dependencies
RUN apt-get update && apt-get install -y \
libcamera-dev \
python3-libcamera \
python3-kms++ \
python3-pyqt5 \
python3-prctl \
libatlas-base-dev \
ffmpeg \
libsm6 \
libxext6 \
&& rm -rf /var/lib/apt/lists/*

RUN pip install picamera2
# Upgrade pip and install essential build tools
RUN pip install --upgrade pip setuptools wheel

# Install the Python dependencies
RUN pip install numpy pandas tqdm gdown opencv-python-headless
RUN pip install flask deepface
RUN pip install tf-keras
RUN pip install tflite-runtime
RUN pip install paho-mqtt



# Copy the rest of your app into the container
COPY . .

# Expose the port (change if needed)
EXPOSE 5000

# Command to run your app
CMD ["python", "pi-camera.py"]
