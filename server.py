import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory, render_template
import shutil
from werkzeug.utils import secure_filename

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

        # Only proceed if it's a file (not a directory)
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


UPLOAD_FOLDER = '/Users/niccolo/Desktop/aTale/aTale/NewFolder'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files[]')
    personInPhoto = request.form.get('description')
    uploaded_files = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_files.append(filename)
        else:
            return jsonify({'error': f'File {file.filename} is not allowed'}), 400

    # Specify the ID of the folder where you want to upload files.
    folder_id = '1TEdvCAaY6aTjwyZ9QZPMc7IjEM92uDHN'

    uploadFolderToDrive(personInPhoto, folder_id, UPLOAD_FOLDER)
    clean_folder(UPLOAD_FOLDER)
    return jsonify({'message': 'Files successfully uploaded', 'files': uploaded_files}), 200
