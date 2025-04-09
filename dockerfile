# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /

RUN apt-get update

# Copy the requirements file and install dependencies
RUN pip install numpy pandas tqdm gdown opencv-python-headless
RUN pip install flask deepface
RUN  pip install tf-keras

RUN pip install tflite-runtime
RUN pip install paho-mqtt
RUN apt install -y libgl1  
RUN apt install -y libglib2.0-0

# Copy the rest of your app
COPY . .

# Expose the port (change if needed)
EXPOSE 5000

# Command to run your app
CMD ["python", "app-mqtt.py"]
