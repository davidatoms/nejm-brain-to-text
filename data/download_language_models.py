#!/usr/bin/env python3
"""
Download 3-gram and 5-gram language models from the separate Dryad dataset.
This script downloads the language models that are not included in the main download_data.py script.

The language models are available at: https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq
- languageModel.tar.gz (3-gram model)
- languageModel_5gram.tar.gz (5-gram model)
"""

import os
import sys
import urllib.request
import json
import tarfile

def display_progress_bar(block_num, block_size, total_size, message=""):
    """Display download progress bar."""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()

def main():
    """Download language models from Dryad."""
    # Language model dataset DOI
    DRYAD_DOI = "10.5061/dryad.x69p8czpq"
    DRYAD_ROOT = "https://datadryad.org"
    
    # Get the list of files from the latest version on Dryad
    urlified_doi = DRYAD_DOI.replace("/", "%2F")
    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    
    print("Fetching dataset information...")
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    # Get the latest version (last in the list)
    latest_version = versions_info["_embedded"]["stash:versions"][-1]
    files_url_path = latest_version["_links"]["stash:files"]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"
    
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]
    
    # Filter for language model files
    language_model_files = []
    for file_info in file_infos:
        filename = file_info["path"]
        if filename in ["languageModel.tar.gz", "languageModel_5gram.tar.gz"]:
            language_model_files.append(file_info)
    
    if not language_model_files:
        print("No language model files found in the dataset.")
        return
    
    print(f"Found {len(language_model_files)} language model files:")
    for file_info in language_model_files:
        print(f"  - {file_info['path']}")
    
    # Download each language model file
    for file_info in language_model_files:
        filename = file_info["path"]
        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"
        download_to_filepath = os.path.join(".", filename)
        
        print(f"\nDownloading {filename}...")
        urllib.request.urlretrieve(
            download_url,
            download_to_filepath,
            reporthook=lambda *args: display_progress_bar(
                *args, message=f"Downloading {filename}"
            ),
        )
        sys.stdout.write("\n")
        
        # Extract tar.gz files
        if filename.endswith(".tar.gz"):
            print(f"Extracting {filename}...")
            with tarfile.open(download_to_filepath, "r:gz") as tar:
                tar.extractall(".")
            print(f"Extracted {filename}")
    
    print("\nLanguage model download complete!")
    print("\nNext steps:")
    print("1. Move the extracted language model directories to ../language_model/pretrained_language_models/")
    print("2. Update your language model paths in evaluation scripts accordingly")

if __name__ == "__main__":
    main()
