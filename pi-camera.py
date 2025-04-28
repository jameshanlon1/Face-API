from flask import Flask, jsonify, send_from_directory
from deepface import DeepFace
import paho.mqtt.client as mqtt
import json
import os
import shutil
import time
from threading import Thread
import io
import base64
from picamera2 import Picamera2

# Try importing picamera - it should be available on Raspberry Pi
try:
    from picamera import PiCamera
    camera_available = True
except ImportError:
    camera_available = False
    print("Warning: PiCamera not available. Camera capture functions will not work.")

app = Flask(__name__)

# MQTT Configuration
MQTT_BROKER = "mqtt.eclipseprojects.io"  # Change to your MQTT broker address
MQTT_PORT = 1883
MQTT_TOPIC = "jamesh/face/verification"

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()  # Start MQTT loop in the background

currentUser = "NONE"
continuous_mode = False
verification_interval = 5  # seconds between verifications in continuous mode

# Ensure user images directory exists
USER_IMAGES_DIR = "./user_images"
if not os.path.exists(USER_IMAGES_DIR):
    os.makedirs(USER_IMAGES_DIR)

# Initialize camera if available
if camera_available:
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 24
    time.sleep(2)  # Allow camera to warm up

def verify_face(image_path):
    global currentUser
    
    returnObj = {"user": "UNKNOWN", "verified": False}
    
    try:
        # Get list of all users
        users = os.listdir(USER_IMAGES_DIR)
        
        # Iterate through each user's face images
        for user in users:
            user_image_dir = os.path.join(USER_IMAGES_DIR, user)
            if os.path.isdir(user_image_dir):
                # Check all images for this user
                user_images = [f for f in os.listdir(user_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for image in user_images:
                    try:
                        user_image_path = os.path.join(user_image_dir, image)
                        result = DeepFace.verify(image_path, user_image_path, model_name="ArcFace", enforce_detection=False)
                        
                        if result["verified"]:
                            currentUser = user
                            returnObj = {"user": currentUser, "verified": True}
                            break  # Found a match, exit the loop
                    except Exception as e:
                        # Continue checking other images if one fails
                        continue
                        
            if returnObj["verified"]:
                break  # Found a match, exit the outer loop
                
        # Publish result to MQTT broker
        mqtt_payload = json.dumps(returnObj)
        mqtt_client.publish(MQTT_TOPIC, mqtt_payload)
        
        return returnObj
        
    except Exception as e:
        error_response = {"user": "UNKNOWN", "verified": False, "error": str(e)}
        mqtt_client.publish(MQTT_TOPIC, json.dumps(error_response))
        return error_response

def continuous_verification():
    """Background thread function for continuous face verification"""
    global continuous_mode
    
    while continuous_mode:
        try:
            if camera_available:
                # Capture image
                image_stream = io.BytesIO()
                camera.capture(image_stream, 'jpeg')
                image_stream.seek(0)
                
                # Save the image
                with open("latest_capture.jpg", "wb") as f:
                    f.write(image_stream.read())
                
                # Verify the face
                result = verify_face("latest_capture.jpg")
                print(f"Continuous verification: {result}")
                
            time.sleep(verification_interval)
        except Exception as e:
            print(f"Error in continuous verification: {e}")
            time.sleep(verification_interval)

@app.route('/')
def index():
    return "Welcome to Raspberry Pi Face Verification Service!"


@app.route('/camera')
def camera_page():
    return send_from_directory('.', 'camera.html')

if __name__ == '__main__':
    # If using the original user "James", migrate their image to the new structure
    if os.path.exists("./images/me1.jpg"):
        james_dir = os.path.join(USER_IMAGES_DIR, "James")
        if not os.path.exists(james_dir):
            os.makedirs(james_dir)
        # Copy the existing James image to the new structure
        if not os.path.exists(os.path.join(james_dir, "James_1.jpg")):
            shutil.copy("./images/me1.jpg", os.path.join(james_dir, "James_1.jpg"))
    
    app.run(host='0.0.0.0', port=5000)