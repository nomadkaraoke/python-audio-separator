#!/usr/bin/env python3

import os
import sys
import json
import hashlib
import requests

MODEL_CACHE_PATH = "/tmp/audio-separator-models"
VR_MODEL_DATA_LOCAL_PATH = f"{MODEL_CACHE_PATH}/vr_model_data.json"
MDX_MODEL_DATA_LOCAL_PATH = f"{MODEL_CACHE_PATH}/mdx_model_data.json"

MODEL_DATA_URL_PREFIX = "https://raw.githubusercontent.com/TRvlvr/application_data/main"
VR_MODEL_DATA_URL = f"{MODEL_DATA_URL_PREFIX}/vr_model_data/model_data_new.json"
MDX_MODEL_DATA_URL = f"{MODEL_DATA_URL_PREFIX}/mdx_model_data/model_data_new.json"

OUTPUT_PATH = f"{MODEL_CACHE_PATH}/model_hashes.json"


def get_model_hash(model_path):
    """
    Get the hash of a model file
    """
    # print(f"Getting hash for model at {model_path}")
    try:
        with open(model_path, "rb") as f:
            f.seek(-10000 * 1024, 2)  # Move the file pointer 10MB before the end of the file
            hash_result = hashlib.md5(f.read()).hexdigest()
            # print(f"Hash for {model_path}: {hash_result}")
            return hash_result
    except IOError:
        with open(model_path, "rb") as f:
            hash_result = hashlib.md5(f.read()).hexdigest()
            # print(f"IOError encountered, hash for {model_path}: {hash_result}")
            return hash_result


def download_file_if_missing(url, local_path):
    """
    Download a file from a URL if it doesn't exist locally
    """
    print(f"Checking if {local_path} needs to be downloaded from {url}")
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}")
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {url} to {local_path}")
    else:
        print(f"{local_path} already exists. Skipping download.")


def load_json_data(file_path):
    """
    Load JSON data from a file
    """
    print(f"Loading JSON data from {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print(f"Loaded JSON data successfully from {file_path}")
            return data
    except FileNotFoundError:
        print(f"{file_path} not found.")
        sys.exit(1)


def iterate_and_hash(directory):
    """
    Iterate through a directory and hash all model files
    """
    print(f"Iterating through directory {directory} to hash model files")
    model_files = [(file, os.path.join(root, file)) for root, _, files in os.walk(directory) for file in files if file.endswith((".pth", ".onnx"))]

    download_file_if_missing(VR_MODEL_DATA_URL, VR_MODEL_DATA_LOCAL_PATH)
    download_file_if_missing(MDX_MODEL_DATA_URL, MDX_MODEL_DATA_LOCAL_PATH)

    vr_model_data = load_json_data(VR_MODEL_DATA_LOCAL_PATH)
    mdx_model_data = load_json_data(MDX_MODEL_DATA_LOCAL_PATH)

    combined_model_params = {
        **vr_model_data,
        **mdx_model_data,
    }

    model_info_list = []
    for file, file_path in sorted(model_files):
        file_hash = get_model_hash(file_path)
        model_info = {
            "file": file,
            "hash": file_hash,
            "params": combined_model_params.get(file_hash, "Parameters not found"),
        }
        model_info_list.append(model_info)

    print(f"Writing model info list to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as json_file:
        json.dump(model_info_list, json_file, indent=4)
        print(f"Successfully wrote model info list to {OUTPUT_PATH}")


if __name__ == "__main__":
    iterate_and_hash(MODEL_CACHE_PATH)
