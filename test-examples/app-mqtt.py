from flask import Flask, request, jsonify
from deepface import DeepFace
import paho.mqtt.client as mqtt
import json

app = Flask(__name__)

# MQTT Configuration
MQTT_BROKER = "test.mosquitto.org"  # Change to your MQTT broker address
MQTT_PORT = 1883
MQTT_TOPIC = "fxwalsh/face/verification"

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()  # Start MQTT loop in the background

userName = "NONE"

@app.route('/verify', methods=['POST'])
def verify():
    global userName
    try:
        file1 = request.files['image1']
        file1.save("face1.jpg")

        returnObj = {"user": "UNKNOWN", "verified": False}

        # Compare faces using DeepFace
        result = DeepFace.verify("face1.jpg", "me2.jpg", model_name="Facenet", enforce_detection=False)

        if result["verified"]:
            userName = "Frank"
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)