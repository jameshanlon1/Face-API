
from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace
import paho.mqtt.client as mqtt
import json
import os
import shutil

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

# Ensure user images directory exists
USER_IMAGES_DIR = "./images"
if not os.path.exists(USER_IMAGES_DIR):
    os.makedirs(USER_IMAGES_DIR)

@app.route('/')
def index():
    return "Welcome to Multi-User Face Verification Service!"  

@app.route('/verify', methods=['POST'])
def verify():
    global currentUser
    try:
        file1 = request.files['image1']
        file1.save("face1.jpg")

        returnObj = {"user": "UNKNOWN", "verified": False}
        
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
                        image_path = os.path.join(user_image_dir, image)
                        result = DeepFace.verify("face1.jpg", image_path, model_name="ArcFace", enforce_detection=False)
                        
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

        return jsonify(returnObj)

    except Exception as e:
        error_response = {"user": "UNKNOWN", "verified": False, "error": str(e)}
        mqtt_client.publish(MQTT_TOPIC, json.dumps(error_response))  # Publish error info to MQTT
        return jsonify(error_response)

@app.route('/user', methods=['GET'])
def user():
    try:
        userObj = {'value': currentUser}
        return jsonify(userObj)  
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/camera')
def camera_page():
    return send_from_directory('.', 'camera.html')        


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)