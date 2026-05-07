import time
import threading
import shutil
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import subprocess
import json
import webbrowser

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'NewFolder')
MATCHING_JSON = os.path.join(BASE_DIR, 'matching.json')
ENCODINGS_PATH = os.path.join(BASE_DIR, 'model', 'encodings.pkl')
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, 'service_key.json')
CREDENTIALS_FILE = os.path.join(BASE_DIR, 'drive_credentials.json')
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, 'client_secrets.json')
DRIVE_FOLDER_ID = '1TEdvCAaY6aTjwyZ9QZPMc7IjEM92uDHN'
ENCODINGS_FILE_ID = '1-0hSQk6Du-NYpjRG5Nm7YSRxBy78vVJP'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_monitor_thread = None
_monitor_lock = threading.Lock()

################### Generic JSON access and update functions
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
    print(f"Added: {name} with path {audio_path}")
###################

# Displays the main page of the server
@app.route('/')
def index():
    return render_template('index.html')

# Allows the user to upload its images on the specified google Drive folder
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files[]')
    personInPhoto = request.form.get('description')
    if not personInPhoto:
        return jsonify({'error': 'description (person name) is required'}), 400
    # Reserve a key in matching.json for this person if they're new
    existing = read_json(MATCHING_JSON)
    if personInPhoto not in existing:
        add_audio_to_person(MATCHING_JSON, personInPhoto, None)
    uploaded_files = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_files.append(filename)
        else:
            return jsonify({'error': f'File {file.filename} is not allowed'}), 400

    uploadFolderToDrive(personInPhoto, DRIVE_FOLDER_ID, UPLOAD_FOLDER)
    clean_folder(UPLOAD_FOLDER)

    return jsonify({'message': 'Files successfully uploaded', 'files': uploaded_files}), 200

################### START UPLOAD TO DRIVE FUNCTIONS

def authenticate_drive():
    gauth = GoogleAuth()
    gauth.settings['client_config_file'] = CLIENT_SECRETS_FILE
    gauth.LoadCredentialsFile(CREDENTIALS_FILE)
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(CREDENTIALS_FILE)
    return GoogleDrive(gauth)

def get_or_create_drive_folder(drive, name, parent_id):
    name = str(name)
    query = (
        f"'{parent_id}' in parents and "
        f"title = '{name}' and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    existing = drive.ListFile({'q': query}).GetList()
    if existing:
        print(f"Reusing existing Drive folder '{name}'.")
        return existing[0]
    folder = drive.CreateFile({
        'title': name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': parent_id}],
    })
    folder.Upload()
    print(f"Created new Drive folder '{name}'.")
    return folder

def uploadFolderToDrive(name, folder_id, local_directory):
    drive = authenticate_drive()
    folder = get_or_create_drive_folder(drive, name, folder_id)

    for filename in os.listdir(local_directory):
        file_path = os.path.join(local_directory, filename)
        if os.path.isfile(file_path):
            gfile = drive.CreateFile({'parents': [{'id': folder['id']}]})
            gfile.SetContentFile(file_path)
            gfile['title'] = filename
            gfile.Upload()
            print(f'Uploaded {filename} to Drive folder {folder["title"]}.')

def clean_folder(folder_path):
    if os.path.exists(folder_path):
        # Iterate over all files and directories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # If it's a file, remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # If it's a directory, remove it and all its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"Folder {folder_path} does not exist!")

################### END UPLOAD TO DRIVE FUNCTIONS

# /notify starts a background watcher that downloads encodings.pkl whenever it
# changes on Drive. Returns immediately so the request thread doesn't block.
@app.route('/notify', methods=['POST'])
def trainModel():
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return jsonify({'message': 'Already monitoring encodings.'}), 200
        service = accessFolderFromDrive()
        _monitor_thread = threading.Thread(
            target=monitor_file,
            args=(ENCODINGS_FILE_ID, ENCODINGS_PATH, service),
            daemon=True,
        )
        _monitor_thread.start()
    return jsonify({'message': 'Encodings monitor started.'}), 202


################### START BRING LOCALLY THE ENCODINGS FUNCTIONS

def accessFolderFromDrive():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Error loading service account credentials: {e}")
        raise

def get_file_metadata(file_id, service):
    return service.files().get(fileId=file_id, fields='modifiedTime').execute()

def monitor_file(file_id, local_path, service, check_interval=60):
    last_modified = None
    while True:
        try:
            metadata = get_file_metadata(file_id, service)
            modified_time = metadata['modifiedTime']
            if modified_time != last_modified:
                print("Encodings modified")
                download_file(file_id, local_path, service)
                last_modified = modified_time
        except Exception as e:
            print(f"monitor_file error: {e}")
        time.sleep(check_interval)

def download_file(file_id, local_path, service):
    request = service.files().get_media(fileId=file_id)
    with open(local_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f'Downloading in local directory: {int(status.progress() * 100)}%.')

################### END BRING LOCALLY ENCODINGS FUNCTIONS


RECORDING_FOLDER = os.path.join(BASE_DIR, 'uploads')
CONVERTED_FOLDER = os.path.join(BASE_DIR, 'recordings')
os.makedirs(RECORDING_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

# Bring the uploaded audio file inside a local directory
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400
    if shutil.which('ffmpeg') is None:
        return jsonify({"error": "ffmpeg not found on PATH"}), 500

    audio = request.files['audio']
    person_name = request.form.get('name')
    if not person_name:
        return jsonify({"error": "name (person) form field is required"}), 400

    safe_name = secure_filename(audio.filename) or 'audio.webm'
    audio_path = os.path.join(RECORDING_FOLDER, safe_name)
    audio.save(audio_path)

    base = os.path.splitext(safe_name)[0]
    output_path = os.path.join(CONVERTED_FOLDER, f"{base}.mp3")

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, '-vn', '-ar', '44100', '-ac', '2', '-ab', '192k', output_path],
            check=True,
        )
        os.remove(audio_path)
        add_audio_to_person(MATCHING_JSON, person_name, output_path)
        return jsonify({"message": "Audio converted to MP3 successfully!", "file": output_path}), 200
    except subprocess.CalledProcessError:
        return jsonify({"error": "Error during audio conversion."}), 500

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5007")
    app.run(debug=True, port=5007, use_reloader=False)