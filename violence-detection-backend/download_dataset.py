import os
import subprocess
import zipfile
from pathlib import Path

def setup_kaggle():
    """setup kaggle API for dataset download"""
    print("🔧 Setting up Kaggle API...")
    
    # check if kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle API already installed")
    except ImportError:
        print("📦 Installing Kaggle API...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
    
    # check for kaggle credentials
    kaggle_dir = Path.home() / ".kaggle"
    credentials_file = kaggle_dir / "kaggle.json"
    
    if not credentials_file.exists():
        print("⚠️  Kaggle credentials not found!")
        print("📝 To download the dataset, you need to:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Create API token (download kaggle.json)")
        print("   3. Place kaggle.json in ~/.kaggle/")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle credentials found")
    return True

def download_dataset():
    """download the DCSASS dataset from kaggle"""
    print("📥 Downloading DCSASS Dataset...")
    
    if not setup_kaggle():
        print("❌ Cannot download without Kaggle credentials")
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # download the dataset
        dataset_name = "mateohervas/dcsass-dataset"
        api.dataset_download_files(dataset_name, path="./data", unzip=True)
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def explore_downloaded_data():
    """explore the downloaded dataset structure"""
    data_dir = Path("./data")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return
    
    print("📁 Dataset Structure:")
    for item in data_dir.iterdir():
        if item.is_dir():
            video_count = len(list(item.glob("*.mp4"))) + len(list(item.glob("*.avi")))
            print(f"  📂 {item.name}: {video_count} videos")
        else:
            print(f"  📄 {item.name}")

if __name__ == "__main__":
    print("🚀 DCSASS Dataset Downloader")
    print("=" * 40)
    
    # check if dataset is already downloaded
    data_dir = Path("./data")
    if data_dir.exists() and any(data_dir.iterdir()):
        print("✅ Dataset already exists!")
        explore_downloaded_data()
    else:
        print("📥 Dataset not found, ready to download...")
        print("💡 Run this script when you have Kaggle credentials set up")
        print("💡 Or manually download from: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset")