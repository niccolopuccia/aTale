import time
import cv2
import numpy as np
import os
from flask import Flask
import pickle
import face_recognition
from PIL import Image

app = Flask(__name__)


# Load precomputed face encodings
with open('/Users/niccolo/Desktop/aTale/aTale/model/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = '/Users/niccolo/Desktop/aTale/aTale/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_classifier = cv2.CascadeClassifier('/Users/niccolo/Desktop/aTale/aTale/model/haarcascade_frontalface_default.xml')

def detect_faces(img, label):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Adding a label above the rectangle
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return img

def predict(captured_frame):
    # Save the captured frame
    if captured_frame is not None:
        filename = 'image.jpg'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, captured_frame)
        print(f"Captured frame saved to {filepath}")

        # Load and preprocess the image
        frame = cv2.imread(filepath)
        if frame is None:
            print({'error': 'Failed to read image'}), 400

        # Resize while maintaining aspect ratio
        height, width = frame.shape[:2]
        new_width = 500
        new_height = int(height * (new_width / width))
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using OpenCV's Haar Cascade
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if detector.empty():
            print({'error': 'Failed to load face detector'}), 500

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert coordinates to the format expected by face_recognition
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # Convert the image to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Compute face encodings
        encodings = face_recognition.face_encodings(rgb, boxes)
        if not encodings:
            print({'error': 'No faces found in the image'}), 400

        face_names = []
        threshold = 0.45  # Default tolerance value

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=threshold)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            if matches:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            face_names.append(name)
            return face_names


cap = cv2.VideoCapture(0)

# Variable to store a captured frame
captured_frame = False
start_time = time.time()
recognized_person = 'Person'

while True:
    ret, frame = cap.read()
    frame = detect_faces(frame, str(recognized_person))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        cv2.imshow('Video Face Detection', frame)
        # Check if 1 second has passed since the start
        if captured_frame is False and time.time() - start_time >= 2:
            print("Automatically capturing a frame after 1 second...")
            captured_frame = frame.copy()
            recognized_person = predict(captured_frame)
            frame_captured = True

            
cap.release()
cv2.destroyAllWindows()