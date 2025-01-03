import os

def create_directory(dir_path):
    if not os.path.exist(dir_path):
        os.makedirs(dir_path)
