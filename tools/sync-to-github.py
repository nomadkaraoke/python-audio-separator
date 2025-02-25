#! /usr/bin/env python3
import os
import requests
import hashlib
from typing import List, Dict
import sys

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()  # Add .strip() to remove whitespace
REPO_OWNER = "nomadkaraoke"
REPO_NAME = "python-audio-separator"
RELEASE_TAG = "model-configs"

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}


def debug_request(url: str, headers: dict, response: requests.Response):
    """Debug helper to print request and response details."""
    print("\n=== Debug Information ===")
    print(f"Request URL: {url}")
    print(f"Request Headers: {headers}")
    print(f"Response Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {response.text[:500]}...")  # First 500 chars of response
    print("=======================\n")


def get_release_assets() -> List[Dict]:
    """Get all assets from the specified release."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{RELEASE_TAG}"
    print(f"\nDebug: Using token: {GITHUB_TOKEN[:4]}...{GITHUB_TOKEN[-4:]}")  # Show first/last 4 chars
    print(f"Debug: Requesting URL: {url}")
    print(f"Debug: Headers: {HEADERS}")

    response = requests.get(url, headers=HEADERS)
    debug_request(url, HEADERS, response)

    if response.status_code != 200:
        print(f"Error getting release: {response.status_code}")
        return []

    release_data = response.json()
    return release_data.get("assets", [])


def list_local_files() -> List[str]:
    """List all files in the current directory (excluding directories)."""
    return [f for f in os.listdir(".") if os.path.isfile(f)]


def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def upload_asset(release_id: int, filepath: str):
    """Upload a file as a release asset."""
    upload_url = f"https://uploads.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/{release_id}/assets"

    filename = os.path.basename(filepath)
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Content-Type": "application/octet-stream"}

    params = {"name": filename}

    with open(filepath, "rb") as f:
        response = requests.post(upload_url, headers=headers, params=params, data=f)

    if response.status_code == 201:
        print(f"Successfully uploaded {filename}")
    else:
        print(f"Failed to upload {filename}: {response.status_code}")
        print(response.text)


def download_asset(asset: Dict):
    """Download a release asset to the local directory."""
    filename = asset["name"]
    download_url = asset["browser_download_url"]

    print(f"Downloading {filename}...")
    response = requests.get(download_url, headers=HEADERS, stream=True)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
    else:
        print(f"Failed to download {filename}: {response.status_code}")


def main():
    if not GITHUB_TOKEN:
        print("Please set GITHUB_TOKEN environment variable")
        sys.exit(1)

    print(f"Debug: Script starting with token length: {len(GITHUB_TOKEN)}")
    print(f"Debug: Token characters: {[ord(c) for c in GITHUB_TOKEN]}")
    print(f"Debug: Token first/last chars: {GITHUB_TOKEN[:4]}...{GITHUB_TOKEN[-4:]}")

    # Get release ID first
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{RELEASE_TAG}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error getting release: {response.status_code}")
        return

    release_id = response.json()["id"]

    # Get existing assets
    existing_assets = get_release_assets()
    existing_filenames = {asset["name"] for asset in existing_assets}

    # Get local files
    local_files = list_local_files()

    print("\nExisting release assets:")
    for asset in existing_assets:
        print(f"- {asset['name']} ({asset['size']} bytes)")

    # Add download option
    print("\nOptions:")
    print("1. Upload new files")
    print("2. Download all missing files")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ")

    if choice == "1":
        # Original upload logic
        files_to_upload = []
        for local_file in local_files:
            if local_file not in existing_filenames:
                print(f"- {local_file}")
                files_to_upload.append(local_file)

        if files_to_upload:
            files_with_size = [(f, os.path.getsize(f)) for f in files_to_upload]
            files_with_size.sort(key=lambda x: x[1])

            print("\nFiles to upload (in order):")
            for file, size in files_with_size:
                print(f"- {file} ({size / 1024 / 1024:.2f} MB)")

            response = input("\nDo you want to upload these files? (y/n): ")
            if response.lower() == "y":
                for file, _ in files_with_size:
                    upload_asset(release_id, file)
        else:
            print("\nNo new files to upload.")

    elif choice == "2":
        # Download missing files
        files_to_download = []
        for asset in existing_assets:
            if asset["name"] not in local_files:
                files_to_download.append(asset)

        if files_to_download:
            print("\nFiles to download:")
            for asset in files_to_download:
                print(f"- {asset['name']} ({asset['size'] / 1024 / 1024:.2f} MB)")

            response = input("\nDo you want to download these files? (y/n): ")
            if response.lower() == "y":
                for asset in files_to_download:
                    download_asset(asset)
        else:
            print("\nNo files to download. Local directory is in sync.")


if __name__ == "__main__":
    main()
