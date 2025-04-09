from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace

app = Flask(__name__)
CORS(app)


@app.route('/verify', methods=['POST'])
def verify():
    try:
        file1 = request.files['image1']
        file1.save("face1.jpg")
        returnObj={"user":"UNKNOWN", "verified":False}
        # Compare faces using a fast model
        result = DeepFace.verify("face1.jpg", "me2.jpg", model_name="Facenet", enforce_detection=False)
        
        if result["verified"]:
            returnObj={"user":"Frank", "verified":True}
        
            

        return jsonify(returnObj)

    except Exception as e:
        return jsonify({"user": "UNKNOWN", "verified": False, "error": str(e)})
    
@app.route('/user', methods=['GET'])
def user():
    try:
        userObj={'value': userName}
        return jsonify(userObj)  
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)