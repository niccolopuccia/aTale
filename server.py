from flask import Flask, jsonify, request, render_template, redirect, url_for
import os
import cv2
import pickle
import face_recognition
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Load precomputed face encodings
with open('/Users/niccolo/Desktop/aTale/aTale/model/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = '/Users/niccolo/Desktop/aTale/aTale/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # Integrity checks
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    # Put the inserted image in the folder uploads in this working directory
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess the image
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({'error': 'Failed to read image'}), 400

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
            return jsonify({'error': 'Failed to load face detector'}), 500

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert coordinates to the format expected by face_recognition
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # Convert the image to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Compute face encodings
        encodings = face_recognition.face_encodings(rgb, boxes)
        if not encodings:
            return jsonify({'error': 'No faces found in the image'}), 400

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
            # print(face_names)

        return jsonify({'recognized_names': face_names}), 200

    return jsonify({'error': 'Invalid file type'}), 400
