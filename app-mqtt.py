from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace
import paho.mqtt.client as mqtt
import json

app = Flask(__name__)

# MQTT Configuration
MQTT_BROKER = "test.mosquitto.org"  # Change to your MQTT broker address
MQTT_PORT = 1883
MQTT_TOPIC = "jamesh/face/verification"

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()  # Start MQTT loop in the background

userName = "NONE"

@app.route('/')
def index():
    return "Welcome to Face Verification Service!"  


@app.route('/verify', methods=['POST'])
def verify():
    global userName
    try:
        file1 = request.files['image1']
        file1.save("face1.jpg")

        returnObj = {"user": "UNKNOWN", "verified": False}

        # Compare faces using DeepFace
        result = DeepFace.verify("face1.jpg", "./images/me1.jpg", model_name="ArcFace", enforce_detection=False)

        if result["verified"]:
            userName = "James"
            returnObj = {"user": userName, "verified": True}

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
        userObj = {'value': userName}
        return jsonify(userObj)  
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/camera')
def camera_page():
    return send_from_directory('.', 'camera.html')        


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)