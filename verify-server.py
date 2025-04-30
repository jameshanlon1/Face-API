from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import json
from flask_cors import CORS
import paho.mqtt.client as mqtt

app = Flask(__name__)
CORS(app)  # Allow requests from other origins (for frontend stuff)

# Folder where we store user face images
USER_IMAGES_DIR = "./images"
os.makedirs(USER_IMAGES_DIR, exist_ok=True)

# MQTT setup (for sending verification results to other devices/services)
MQTT_BROKER = "mqtt.eclipseprojects.io"
MQTT_PORT = 1883
MQTT_TOPIC = "jamesh/face/verification"

# Connect to MQTT broker
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

@app.route('/verify', methods=['POST'])
def verify():
    """
    Compares uploaded face with stored faces to see if there's a match.
    If matched, sends result over MQTT and returns the info as JSON.
    """
    try:
        # Save uploaded image temporarily
        file1 = request.files['image1']
        file1.save("face1.jpg")

        result_data = {"user": "UNKNOWN", "verified": False}

        # Check each user's folder in the images directory
        for user in os.listdir(USER_IMAGES_DIR):
            user_dir = os.path.join(USER_IMAGES_DIR, user)

            if os.path.isdir(user_dir):
                # Get all image files for that user
                user_images = [img for img in os.listdir(user_dir)
                               if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

                for img in user_images:
                    try:
                        img_path = os.path.join(user_dir, img)

                        # Use DeepFace to compare uploaded image with stored image
                        result = DeepFace.verify("face1.jpg", img_path,
                                                 model_name="Facenet", enforce_detection=False)

                        if result["verified"]:
                            result_data["user"] = user
                            result_data["verified"] = True
                            break  # Found a match, no need to keep checking
                    except:
                        # If something fails (like a bad image), skip it
                        continue

            if result_data["verified"]:
                break  # Already found a match, exit outer loop too

        # Send result via MQTT
        mqtt_client.publish(MQTT_TOPIC, json.dumps(result_data))

        return jsonify(result_data)

    except Exception as e:
        # Handle any errors that occur and send that too
        error_data = {"user": "UNKNOWN", "verified": False, "error": str(e)}
        mqtt_client.publish(MQTT_TOPIC, json.dumps(error_data))
        return jsonify(error_data)

if __name__ == '__main__':
    print("Loading Facenet model... this might take a second.")
    DeepFace.build_model("Facenet")  # Load once at startup so it's fast later
    app.run(host='0.0.0.0', port=5000)
