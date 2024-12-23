import shutil
import time
import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, send_file, request
import pickle
import face_recognition

app = Flask(__name__)

# Load precomputed face encodings
with open('/Users/niccolo/Desktop/aTale/aTale/model/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']

# Configure upload folder
UPLOAD_FOLDER = '/Users/niccolo/Desktop/aTale/aTale/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Directory to store sound files
SOUNDS_FOLDER = '/Users/niccolo/Desktop/aTale/aTale/recordings'
os.makedirs(SOUNDS_FOLDER, exist_ok=True)

# Load Haar Cascade
face_classifier = cv2.CascadeClassifier('/Users/niccolo/Desktop/aTale/aTale/model/haarcascade_frontalface_default.xml')

def detect_faces(img, label="Person"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is None or len(faces) == 0:
        return img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return img

def save_frame(frame, i):
    if frame is not None:
        filename = f'image{i}.jpg'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    return None

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

def predict(filepath):
    frame = cv2.imread(filepath)
    if frame is None:
        return "Unknown"

    # Resize frame
    height, width = frame.shape[:2]
    new_width = 500
    new_height = int(height * (new_width / width))
    frame = cv2.resize(frame, (new_width, new_height))

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    face_names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        if matches and len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        face_names.append(name)

    return ", ".join(face_names) if face_names else "Unknown"

face_to_sound = {
    "Nicco": "/Users/niccolo/Desktop/aTale/aTale/recordings/Audio1.mp3",
}


@app.route('/')
def index():
    return render_template('index2.html')

camera = cv2.VideoCapture(0)
recognized_person = "Unknown"


def generate_frames():
    global recognized_person
    start_time = time.time()
    i = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perform face detection and recognition every 2 seconds
        if time.time() - start_time >= 2:
            filepath = save_frame(frame, i)
            if filepath:
                recognized_person = predict(filepath)
            start_time = time.time()
            i += 1

        # Draw face detections with the label
        frame = detect_faces(frame, recognized_person)

        # Encode frame as JPEG and yield to the client
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Clear folder every 10 iterations to reduce disruption
        if i % 5 == 0:
            clear_folder(UPLOAD_FOLDER)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_recognition')
def current_recognition():
    return recognized_person

@app.route('/play_sound/<string:name>')
def play_sound(name):
    sound_file = face_to_sound.get(name)
    return send_file(sound_file, mimetype='audio/mpeg')