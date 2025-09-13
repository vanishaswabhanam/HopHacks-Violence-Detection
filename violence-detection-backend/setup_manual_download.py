#!/usr/bin/env python3
"""
Helper script to set up manually downloaded DCSASS dataset
"""

import os
import zipfile
import shutil
from pathlib import Path

def setup_manual_download():
    """help user set up manually downloaded dataset"""
    print("📁 Manual Dataset Setup Helper")
    print("=" * 40)
    
    # check if data folder exists
    data_dir = Path("./data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("✅ Created data directory")
    
    # look for zip files in current directory
    zip_files = list(Path(".").glob("*.zip"))
    
    if not zip_files:
        print("❌ No zip files found in current directory")
        print("📝 Please:")
        print("   1. Download the dataset zip file from Kaggle")
        print("   2. Place it in this directory (violence-detection-backend)")
        print("   3. Run this script again")
        return False
    
    print(f"📦 Found zip files: {[f.name for f in zip_files]}")
    
    # find the DCSASS dataset zip
    dcsass_zip = None
    for zip_file in zip_files:
        if ("dcsass" in zip_file.name.lower() or 
            "dataset" in zip_file.name.lower() or 
            "archive" in zip_file.name.lower()):
            dcsass_zip = zip_file
            break
    
    if not dcsass_zip:
        print("⚠️  Could not identify DCSASS dataset zip file")
        print("💡 Please rename your zip file to contain 'dcsass' or 'dataset'")
        return False
    
    print(f"🎯 Found DCSASS dataset: {dcsass_zip.name}")
    
    # extract the zip file
    try:
        print("📤 Extracting dataset...")
        with zipfile.ZipFile(dcsass_zip, 'r') as zip_ref:
            zip_ref.extractall("./data")
        print("✅ Dataset extracted successfully!")
        
        # verify extraction
        verify_extraction()
        
        # clean up zip file (optional)
        print(f"🗑️  Would you like to delete the zip file? ({dcsass_zip.name})")
        print("💡 You can delete it to save space, or keep it as backup")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False

def verify_extraction():
    """verify the dataset was extracted correctly"""
    data_dir = Path("./data")
    
    # look for expected folders
    expected_folders = ["Fighting", "Robbery", "Abuse", "Arrest", "Arson", "Assault", 
                       "Burglary", "Explosion", "RoadAccidents", "Shooting", 
                       "Shoplifting", "Stealing", "Vandalism"]
    
    found_folders = []
    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name in expected_folders:
            found_folders.append(folder.name)
    
    print(f"\n🔍 Dataset Verification:")
    print(f"📁 Found {len(found_folders)} categories:")
    
    total_videos = 0
    for folder in sorted(found_folders):
        video_count = len(list((data_dir / folder).glob("*.mp4"))) + len(list((data_dir / folder).glob("*.avi")))
        total_videos += video_count
        print(f"   {folder}: {video_count} videos")
    
    print(f"\n📊 Total videos found: {total_videos}")
    
    if len(found_folders) >= 10:
        print("✅ Dataset looks good!")
        return True
    else:
        print("⚠️  Dataset might be incomplete")
        return False

if __name__ == "__main__":
    if setup_manual_download():
        print("\n🎉 Dataset ready for processing!")
        print("🚀 Next step: Run video processing pipeline")
    else:
        print("\n❌ Setup failed")
        print("💡 Make sure you downloaded the zip file to this directory")