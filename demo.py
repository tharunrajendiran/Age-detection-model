import os
import cv2
from src.face_detector import detect_faces
from src.age_estimator import predict_age
from src.gender_classifier import predict_gender

def process_images(folder_path="data/"):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        faces = detect_faces(image)
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            age = predict_age(face)
            gender = predict_gender(face)
            print(f"{filename} → Gender: {gender}, Age: {age}")

if __name__ == "__main__":
    process_images()
