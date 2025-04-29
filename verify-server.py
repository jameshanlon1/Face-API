from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import json

app = Flask(__name__)

USER_IMAGES_DIR = "./images"
os.makedirs(USER_IMAGES_DIR, exist_ok=True)

@app.route('/verify', methods=['POST'])
def verify():
    try:
        file = request.files['image1']
        file.save("face1.jpg")

        result_obj = {"user": "UNKNOWN", "verified": False}

        users = os.listdir(USER_IMAGES_DIR)
        for user in users:
            user_dir = os.path.join(USER_IMAGES_DIR, user)
            if not os.path.isdir(user_dir):
                continue

            user_images = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for image in user_images:
                try:
                    image_path = os.path.join(user_dir, image)
                    print(f"🔍 Comparing with: {image_path}")
                    result = DeepFace.verify(
                        "face1.jpg", image_path,
                        model_name="Facenet",
                        enforce_detection=True
                    )
                    print(f"✅ Result: verified={result['verified']}, distance={result.get('distance')}")

                    if result.get("verified") and result.get("distance", 1.0) < 0.55:
                        result_obj = {"user": user, "verified": True}
                        break

                except Exception as e:
                    print(f"❌ Failed comparing with {image}: {e}")
                    continue

            if result_obj["verified"]:
                break

        return jsonify(result_obj)

    except Exception as e:
        print(f"🔥 CRASH in /verify: {e}")
        return jsonify({"user": "UNKNOWN", "verified": False, "error": str(e)})

if __name__ == '__main__':
    from deepface import DeepFace
    print("🔧 Preloading model (Facenet)...")
    DeepFace.build_model("Facenet")  # avoid load time on first request
    app.run(host='0.0.0.0', port=8000)
