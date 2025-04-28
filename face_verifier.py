#!/usr/bin/env python3

import os, time, json, logging, threading
import numpy as np
from deepface import DeepFace
import paho.mqtt.client as mqtt
from picamera2 import Picamera2

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MQTT_BROKER, MQTT_PORT = "mqtt.eclipseprojects.io", 1883
MQTT_TOPIC = "jamesh/face/verification"
USER_IMAGES_DIR = "./images"
INTERVAL = 2  # Faster!
THREADING = True

# Globals
model = None
user_embeddings = {}  # {username: [embeddings]}
verification_in_progress = False

# MQTT setup
def setup_mqtt():
    client = mqtt.Client(client_id=f"pi-face-verifier-{int(time.time())}")
    client.on_connect = lambda c, u, f, rc: logging.info("Connected to MQTT" if rc == 0 else f"MQTT connect failed: {rc}")
    client.will_set(f"{MQTT_TOPIC}/status", json.dumps({"status": "offline"}), qos=1, retain=True)
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        client.publish(f"{MQTT_TOPIC}/status", json.dumps({"status": "online"}))
        return client
    except Exception as e:
        logging.error(f"MQTT init failed: {e}")
        return None

# Camera setup
def setup_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        time.sleep(1)
        logging.info("Camera initialized")
        return picam2
    except Exception as e:
        logging.error(f"Camera init error: {e}")
        return None

# Load model once
def preload_model():
    global model
    model = DeepFace.build_model("ArcFace")
    logging.info("DeepFace model preloaded (ArcFace)")

# Precompute user embeddings
def compute_user_embeddings():
    global user_embeddings
    user_embeddings.clear()

    for user in os.listdir(USER_IMAGES_DIR):
        user_dir = os.path.join(USER_IMAGES_DIR, user)
        if not os.path.isdir(user_dir):
            continue
        
        embeddings = []
        for img_file in os.listdir(user_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(user_dir, img_file)
                try:
                    embedding = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
                    embeddings.append(np.array(embedding))
                except Exception as e:
                    logging.warning(f"Failed to process {img_path}: {e}")
        
        if embeddings:
            user_embeddings[user] = embeddings
            logging.info(f"Loaded {len(embeddings)} embeddings for user '{user}'")
    logging.info(f"User embeddings loaded: {list(user_embeddings.keys())}")

# Capture image to memory
def capture_image(camera):
    if not camera:
        return None
    try:
        frame = camera.capture_array()
        return frame
    except Exception as e:
        logging.error(f"Capture error: {e}")
        return None

# Verify face by comparing embeddings
def verify_face(image_array):
    if image_array is None:
        return {"user": "UNKNOWN", "verified": False}

    try:
        start_time = time.time()

        # Get embedding for captured face
        representations = DeepFace.represent(img_path=image_array, model_name="ArcFace", enforce_detection=False)
        if not representations:
            logging.info("No face detected")
            return {"user": "UNKNOWN", "verified": False}

        captured_embedding = np.array(representations[0]["embedding"])
        best_match = {"user": "UNKNOWN", "verified": False, "distance": float('inf')}

        # Compare with known embeddings
        for user, embeddings in user_embeddings.items():
            for ref_embedding in embeddings:
                distance = np.linalg.norm(captured_embedding - ref_embedding)
                if distance < best_match["distance"]:
                    best_match = {"user": user, "verified": True, "distance": distance}

        # Threshold
        THRESHOLD = 0.7
        if best_match["verified"] and best_match["distance"] < THRESHOLD:
            result = {"user": best_match["user"], "verified": True, "confidence": best_match["distance"]}
        else:
            result = {"user": "UNKNOWN", "verified": False}
        
        logging.info(f"Verification took {time.time() - start_time:.2f}s")
        return result

    except Exception as e:
        logging.error(f"Verification error: {e}")
        return {"user": "UNKNOWN", "verified": False, "error": str(e)}

# MQTT publish
def publish_result(client, result):
    if not client:
        return
    try:
        client.publish(MQTT_TOPIC, json.dumps(result))
        logging.info(f"Published: {result}")
    except Exception as e:
        logging.error(f"MQTT publish error: {e}")

# Threaded verification
def process_verification(mqtt_client, frame):
    global verification_in_progress
    try:
        verification_in_progress = True
        result = verify_face(frame)
        publish_result(mqtt_client, result)
    finally:
        verification_in_progress = False

# Main loop
def main():
    import shutil

    mqtt_client = setup_mqtt()
    camera = setup_camera()
    if not camera:
        return

    preload_model()
    compute_user_embeddings()

    # Legacy image migration
    if os.path.exists("./images/me1.jpg"):
        james_dir = os.path.join(USER_IMAGES_DIR, "James")
        os.makedirs(james_dir, exist_ok=True)
        if not os.path.exists(os.path.join(james_dir, "face.jpg")):
            shutil.copy("./images/me1.jpg", os.path.join(james_dir, "face.jpg"))
            logging.info("Migrated James's image from old location")

    logging.info(f"Starting verification loop every {INTERVAL}s")
    error_count = 0

    try:
        while True:
            try:
                if THREADING and verification_in_progress:
                    time.sleep(0.1)
                    continue

                frame = capture_image(camera)
                if frame is not None:
                    if THREADING:
                        threading.Thread(
                            target=process_verification,
                            args=(mqtt_client, frame),
                            daemon=True
                        ).start()
                    else:
                        result = verify_face(frame)
                        publish_result(mqtt_client, result)
                    error_count = 0
                else:
                    error_count += 1
                    if error_count >= 5:
                        logging.warning("Capture failures, reinitializing camera")
                        camera.close()
                        camera = setup_camera()
                        error_count = 0

            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                error_count += 1
                if error_count >= 10:
                    logging.critical("Too many errors, restarting camera")
                    camera.close()
                    camera = setup_camera()
                    error_count = 0

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        if camera:
            camera.close()
        if mqtt_client:
            mqtt_client.publish(f"{MQTT_TOPIC}/status", json.dumps({"status": "offline"}))
            mqtt_client.disconnect()
        logging.info("Service shutdown complete")

if __name__ == "__main__":
    main()
