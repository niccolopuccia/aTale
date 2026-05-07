import shutil
import time
import threading
import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, send_file
import pickle
import json
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from keras._tf_keras.keras.applications import MobileNet
import webbrowser

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATCHING_JSON = os.path.join(BASE_DIR, 'matching_obj.json')
ENCODINGS_PATH = os.path.join(BASE_DIR, 'model', 'objencodings.pkl')

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

# Build the MobileNet feature extractor once
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, pooling='avg')
net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, 'model', 'deploy.prototxt'),
    os.path.join(BASE_DIR, 'model', 'mobilenet_iter_73000.caffemodel'),
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_obj')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SOUNDS_FOLDER = os.path.join(BASE_DIR, 'recordings_obj')
os.makedirs(SOUNDS_FOLDER, exist_ok=True)

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


def extract_embedding(obj):
    # Verifica se l'oggetto è vuoto
    if obj is None or obj.size == 0:
        print("Oggetto vuoto, non posso estrarre l'embedding.")
        return None

    # Pre-elaborazione dell'immagine
    img = cv2.resize(obj, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Ottieni l'embedding
    embedding = base_model.predict(img)
    return embedding.flatten()


def predict(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return "Unknown"

    with open(ENCODINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        encodings = data['encodings']
        names = data['names']
    embedding = extract_embedding(img)

    # Confronta l'embedding estratto con quelli esistenti
    similarities = cosine_similarity([embedding], encodings)

    # Trova l'indice dell'embedding più simile
    best_match_index = np.argmax(similarities)

    # Ottieni il nome dell'etichetta corrispondente
    best_match_name = names[best_match_index]
    best_match_similarity = similarities[0][best_match_index]

    if best_match_similarity > 0.6:
        print(f' Match migliore: {best_match_name} 'f'con somiglianza: {best_match_similarity:.2f}')
        return best_match_name
    else:
        print("Nessun oggetto con match rilevato nell'immagine.")
        return "Unknown object"



@app.route('/')
def index():
    return render_template('userObjClient.html')

camera = cv2.VideoCapture(0)
_state_lock = threading.Lock()
recognized_item = "Unknown"

def release_camera():
    if camera.isOpened():
        camera.release()

def generate_frames():
    global recognized_item
    start_time = time.time()
    i = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        if time.time() - start_time >= 2:
            filepath = save_frame(frame, i)
            if filepath:
                with _state_lock:
                    recognized_item = predict(filepath)
            start_time = time.time()
            i += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if i % 5 == 0:
            clear_folder(UPLOAD_FOLDER)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_recognition')
def current_recognition():
    with _state_lock:
        return recognized_item

@app.route('/play_sound/<string:name>')
def play_sound(name):
    data = read_json(MATCHING_JSON)
    sound_file = data.get(name)
    if sound_file and os.path.isfile(sound_file):
        return send_file(sound_file, mimetype='audio/mpeg')
    return "Sound file not found", 404

if __name__ == '__main__':
    try:
        webbrowser.open("http://127.0.0.1:5003")
        app.run(debug=True, port=5003, use_reloader=False)
    finally:
        release_camera()