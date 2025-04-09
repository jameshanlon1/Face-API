import cv2
from deepface import DeepFace

def crop_face(image_path):
    image = cv2.imread(image_path)
    x, y, w, h = 50, 50, 200, 200  # Replace with actual coordinates
    face = image[y:y+h, x:x+w]
    cropped_path = "cropped_" + image_path
    cv2.imwrite(cropped_path, face)
    return cropped_path

# Pre-crop images before comparison
img1_cropped = crop_face("me1.jpg")
img2_cropped = crop_face("me2.jpg")

# Compare faces
result = DeepFace.verify(img1_cropped, img2_cropped, model_name="Facenet", enforce_detection=False)

print(result)