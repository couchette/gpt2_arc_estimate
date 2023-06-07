import zipfile
import os


def unzip_file(zip_filepath, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)
