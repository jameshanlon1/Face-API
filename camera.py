from deepface import DeepFace
result = DeepFace.verify("face1.jpg", "./images/James/face1.jpg", model_name="ArcFace", enforce_detection=True)
print(result)
