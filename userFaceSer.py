import shutil
import time
import threading
import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, send_file
import pickle
import face_recognition
import json
import webbrowser

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATCHING_JSON = os.path.join(BASE_DIR, 'matching.json')

# Generic JSON access and update functions
def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def add_audio_to_person(file_path, name, audio_path):
    data = read_json(file_path)
    data[name] = audio_path
    write_json(data, file_path)
    print(f"Aggiunto: {name} con path {audio_path}")

# Load precomputed face encodings
with open(os.path.join(BASE_DIR, 'model', 'encodings.pkl'), 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']

# Configure upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Directory to store sound files
SOUNDS_FOLDER = os.path.join(BASE_DIR, 'recordings')
os.makedirs(SOUNDS_FOLDER, exist_ok=True)

RESIZE_WIDTH = 500

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

def recognize_faces(frame):
    """Return (boxes, names) on the original frame's coordinate system."""
    if frame is None:
        return [], []

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    small = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    small_boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, small_boxes)

    names = []
    for encoding in encodings:
        name = "Unknown"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            if len(distances) > 0:
                best = int(np.argmin(distances))
                if distances[best] <= 0.45:
                    name = known_names[best]
        names.append(name)

    # Map small-frame boxes back to the original frame coordinates
    inv = 1.0 / scale
    boxes = [
        (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
        for (top, right, bottom, left) in small_boxes
    ]
    return boxes, names

def draw_overlays(frame, boxes, names):
    for (top, right, bottom, left), name in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (158, 186, 0), 2)
        cv2.putText(frame, name, (left, max(0, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (158, 186, 0), 2)
    return frame

@app.route('/')
def index():
    return render_template('index2.html')

camera = cv2.VideoCapture(0)
_state_lock = threading.Lock()
_last_boxes = []
_last_names = []
recognized_person = "Unknown"

clear_folder(UPLOAD_FOLDER)

def release_camera():
    if camera.isOpened():
        camera.release()

def generate_frames():
    global recognized_person, _last_boxes, _last_names
    last_recognition = 0.0

    while True:
        success, frame = camera.read()
        if not success:
            break

        if time.time() - last_recognition >= 2:
            boxes, names = recognize_faces(frame)
            with _state_lock:
                _last_boxes, _last_names = boxes, names
                recognized_person = ", ".join(names) if names else "Unknown"
            last_recognition = time.time()

        with _state_lock:
            boxes, names = list(_last_boxes), list(_last_names)
        draw_overlays(frame, boxes, names)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_recognition')
def current_recognition():
    with _state_lock:
        return recognized_person

@app.route('/play_sound/<string:name>')
def play_sound(name):
    data = read_json(MATCHING_JSON)
    sound_file = data.get(name)
    if sound_file and os.path.isfile(sound_file):
        return send_file(sound_file, mimetype='audio/mpeg')
    return "Sound file not found", 404

if __name__ == '__main__':
    try:
        webbrowser.open("http://127.0.0.1:5002")
        app.run(debug=True, port=5002, use_reloader=False)
    finally:
        release_camera()
