#!/usr/bin/env python3
"""
Quick script to download DCSASS dataset once Kaggle credentials are set up
"""

import os
import subprocess
from pathlib import Path

def check_kaggle_credentials():
    """check if kaggle credentials are properly set up"""
    kaggle_file = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_file.exists():
        print("❌ kaggle.json not found!")
        print("📝 Please:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Create API Token")
        print("   3. Download kaggle.json")
        print("   4. Place it in ~/.kaggle/kaggle.json")
        return False
    
    # check file permissions
    stat = kaggle_file.stat()
    if stat.st_mode & 0o777 != 0o600:
        print("🔧 Setting correct permissions...")
        os.chmod(kaggle_file, 0o600)
    
    print("✅ Kaggle credentials found!")
    return True

def download_dataset():
    """download the DCSASS dataset"""
    if not check_kaggle_credentials():
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print("📥 Downloading DCSASS Dataset...")
        print("⏳ This may take a few minutes (768MB)...")
        
        # download dataset
        dataset_name = "mateohervas/dcsass-dataset"
        api.dataset_download_files(dataset_name, path="./data", unzip=True)
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def verify_download():
    """verify the dataset was downloaded correctly"""
    data_dir = Path("./data")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    # check for expected folders
    expected_folders = ["Fighting", "Robbery", "Abuse", "Arrest", "Arson", "Assault", 
                       "Burglary", "Explosion", "RoadAccidents", "Shooting", 
                       "Shoplifting", "Stealing", "Vandalism"]
    
    found_folders = []
    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name in expected_folders:
            found_folders.append(folder.name)
    
    print(f"📁 Found {len(found_folders)} categories:")
    for folder in sorted(found_folders):
        video_count = len(list((data_dir / folder).glob("*.mp4"))) + len(list((data_dir / folder).glob("*.avi")))
        print(f"   {folder}: {video_count} videos")
    
    return len(found_folders) > 0

if __name__ == "__main__":
    print("🚀 DCSASS Dataset Downloader")
    print("=" * 40)
    
    if download_dataset():
        print("\n🔍 Verifying download...")
        if verify_download():
            print("\n🎉 Dataset ready for processing!")
        else:
            print("\n⚠️  Download verification failed")
    else:
        print("\n❌ Download failed")