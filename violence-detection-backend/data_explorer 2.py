import os
import cv2
import json
from pathlib import Path
import numpy as np

def explore_dataset():
    """explore the DCSASS dataset structure and get basic info"""
    
    # we'll simulate the dataset structure since we don't have it downloaded yet
    # but this shows what we'll do when we get it
    
    print("🔍 DCSASS Dataset Explorer")
    print("=" * 50)
    
    # dataset structure based on the kaggle info
    categories = {
        "Abuse": 39,
        "Arrest": 26, 
        "Arson": 22,
        "Assault": 23,
        "Burglary": 47,
        "Explosion": 23,
        "Fighting": 8,  # this is our main focus!
        "RoadAccidents": 76,
        "Robbery": 105,
        "Shooting": 30,
        "Shoplifting": 28,
        "Stealing": 64,
        "Vandalism": 29
    }
    
    print("📊 Dataset Categories:")
    for category, count in categories.items():
        print(f"  {category}: {count} videos")
    
    print(f"\n📈 Total Videos: {sum(categories.values())}")
    print(f"📈 Normal Videos: 9,676")
    print(f"📈 Abnormal Videos: 7,177")
    
    print(f"\n🎯 Key Insights:")
    print(f"  • Fighting category has only {categories['Fighting']} videos")
    print(f"  • We'll need data augmentation for fighting detection")
    print(f"  • Robbery has most videos ({categories['Robbery']}) - good for training")
    
    return categories

def analyze_video_sample(video_path):
    """analyze a single video file to understand its properties"""
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    
    # get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    video_info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }
    
    print(f"📹 Video Analysis:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Total Frames: {frame_count}")
    
    return video_info

if __name__ == "__main__":
    # explore the dataset structure
    categories = explore_dataset()
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Download DCSASS dataset from Kaggle")
    print(f"  2. Extract videos to data/ folder")
    print(f"  3. Analyze sample videos from each category")
    print(f"  4. Create training/validation splits")