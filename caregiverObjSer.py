import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from flask import Flask, request, redirect, url_for, jsonify, render_template
import shutil
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from flask import Flask, request, jsonify
import subprocess
import os
import json
import webbrowser

app = Flask(__name__)
UPLOAD_FOLDER = '/Users/niccolo/Desktop/aTale/aTale/NewFolderObj'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    return render_template('caregiverObjClient.html')

# Allows the user to upload its images on the specified google Drive folder
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files[]')
    personInPhoto = request.form.get('description')
    # Insert a new key in the JSON dictionary with just the name of the person framed
    add_audio_to_person("/Users/niccolo/Desktop/aTale/aTale/matching_obj.json", personInPhoto, None)
    uploaded_files = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_files.append(filename)
        else:
            return jsonify({'error': f'File {file.filename} is not allowed'}), 400

    # Specify the ID of the folder where you want to upload files.
    folder_id = '1cNJVO7VZvkZg4eLhSfjms1g549sKt3o2'

    uploadFolderToDrive(personInPhoto, folder_id, UPLOAD_FOLDER)
    clean_folder(UPLOAD_FOLDER)

    return jsonify({'message': 'Files successfully uploaded', 'files': uploaded_files}), 200

################### START UPLOAD TO DRIVE FUNCTIONS

def uploadFolderToDrive(name, folder_id, local_directory):
    # Authenticate the client.
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver and automatically handles authentication.
    drive = GoogleDrive(gauth)

    # Create a new folder inside the specified Google Drive folder
    folder_name = str(name)
    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': folder_id}]
    }
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()  # Upload the new folder to Google Drive
    print(f"Created new folder '{folder_name}' inside the specified folder.")

    # Iterate over all files in the local directory.
    for filename in os.listdir(local_directory):
        file_path = os.path.join(local_directory, filename)
        print(file_path)

        # Only proceed if it's a file
        if os.path.isfile(file_path):
            # Create a file instance and set its content and parent folder (the newly created folder)
            gfile = drive.CreateFile({'parents': [{'id': folder['id']}]})
            gfile.SetContentFile(file_path)
            gfile['title'] = filename
            gfile.Upload()  # Upload the file
            print(f'Uploaded {filename} to Google Drive inside folder {folder_name}.')

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

# Whenever the retrain button is clicked triggers the encodigs file monitoring and eventually bring it into the local folder when it changes
@app.route('/notify', methods=['POST'])
def trainModel():
    FILE_ID = '1u1XFbqCXgjB_3QVkloONy8b7K9kqbafM'
    LOCAL_PATH = '/Users/niccolo/Desktop/aTale/aTale/model/objencodings.pkl'
    service = accessFolderFromDrive()
    monitor_file(FILE_ID, LOCAL_PATH, service)


################### START BRING LOCALLY THE ENCODINGS FUNCTIONS

def accessFolderFromDrive():
    # Authenticate and Build the Drive Service
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = '/Users/niccolo/Desktop/aTale/aTale/service_key.json'

    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        print(f"Error loading service account credentials: {e}")
        raise e

def get_file_metadata(file_id, service):
    return service.files().get(fileId=file_id, fields='modifiedTime').execute()

def monitor_file(file_id, local_path, service, check_interval=60):
    last_modified = None
    while True:
        metadata = get_file_metadata(file_id, service)
        modified_time = metadata['modifiedTime']
        if modified_time != last_modified:
            print("Encodings modified")
            download_file(file_id, local_path, service)
            last_modified = modified_time
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


# Create a directory to store the uploaded files
RECORDING_FOLDER = 'uploads_obj'
CONVERTED_FOLDER = 'recordings_obj'
os.makedirs(RECORDING_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

# Bring the uploaded audio file inside a local directory
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    # Get the uploaded file
    audio = request.files['audio']
    audio_path = os.path.join(RECORDING_FOLDER, audio.filename)
    audio.save(audio_path)

    # Convert the audio from WEBM to MP3 using ffmpeg
    output_path = os.path.join(CONVERTED_FOLDER, f"{audio.filename.split('.')[0]}.mp3")

    try:
        # Use ffmpeg to convert the audio file to MP3
        subprocess.run(['ffmpeg', '-i', audio_path, '-vn', '-ar', '44100', '-ac', '2', '-ab', '192k', output_path], check=True)
        # Delete the original WEBM file after conversion
        os.remove(audio_path)
        # Look for the last inserted key in the dictionary (the one with a None value) and add the audio path as key
        data = read_json("/Users/niccolo/Desktop/aTale/aTale/matching_obj.json")
        last_key = list(data.keys())[-1]
        add_audio_to_person("/Users/niccolo/Desktop/aTale/aTale/matching_obj.json", last_key, "/Users/niccolo/Desktop/aTale/aTale/" + output_path)
        return jsonify({"message": "Audio converted to MP3 successfully!", "file": output_path}), 200
    except subprocess.CalledProcessError:
        return jsonify({"error": "Error during audio conversion."}), 500
    

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5001")
    app.run(debug=True, port=5001)