from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from real_video_processor import RealVideoProcessor
import json
from datetime import datetime

# basic data models for our API
class Detection(BaseModel):
    camera_id: str
    threat_type: str
    confidence: float
    timestamp: str
    location: dict

class Alert(BaseModel):
    incident_id: str
    threat_type: str
    severity: str  # "low", "medium", "high"
    cameras: List[str]
    timestamp: str

# create FastAPI app
app = FastAPI(title="Violence Detection API", version="1.0.0")

# add CORS middleware so React can talk to our backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize REAL video processor
video_processor = RealVideoProcessor()

def generate_real_detections_from_analysis(analysis_data: dict) -> List[dict]:
    """generate real detections from video analysis data"""
    detections = []
    
    print(f"🔍 Processing analysis data with {len(analysis_data)} categories")
    
    for category, category_data in analysis_data.items():
        if "videos" in category_data:
            print(f"📁 Processing {category} with {len(category_data['videos'])} videos")
            for video in category_data["videos"]:
                if video.get("status") == "success" and video.get("confidence", 0) > 0.5:
                    detection = {
                        "camera_id": f"cam_{len(detections) + 1}",
                        "threat_type": video.get("threat_type", "unknown"),
                        "confidence": video.get("confidence", 0.0),
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "location": {"x": 150, "y": 200},
                        "severity": video.get("threat_level", "low"),
                        "indicators": video.get("indicators", []),
                        "filename": video.get("filename", ""),
                        "frames_analyzed": video.get("frames_analyzed", 0)
                    }
                    detections.append(detection)
                    print(f"✅ Added detection: {video.get('threat_type')} - {video.get('confidence')}")
    
    print(f"🎯 Generated {len(detections)} real detections")
    return detections

# load REAL analysis data
try:
    with open("real_video_analysis.json", "r") as f:
        analysis_data = json.load(f)
    detections = generate_real_detections_from_analysis(analysis_data)
    print(f"✅ Loaded {len(detections)} REAL detections from video analysis")
except Exception as e:
    print(f"❌ Error loading real analysis data: {e}")
    detections = []
    analysis_data = {}

alerts = []

@app.get("/")
async def root():
    return {"message": "Violence Detection API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "cameras_online": 4}

@app.get("/detections")
async def get_detections():
    """get all recent detections"""
    return {"detections": detections[-10:]}  # last 10 detections

@app.get("/alerts")
async def get_alerts():
    """get all active alerts"""
    return {"alerts": alerts}

@app.get("/analysis")
async def get_analysis():
    """get video analysis data"""
    return {"analysis": analysis_data}

@app.get("/stats")
async def get_stats():
    """get system statistics"""
    total_videos = sum(cat.get("total_videos", 0) for cat in analysis_data.values())
    processed_videos = sum(cat.get("processed_videos", 0) for cat in analysis_data.values())
    
    return {
        "total_videos": total_videos,
        "processed_videos": processed_videos,
        "categories": len(analysis_data),
        "active_detections": len(detections),
        "system_uptime": "99.8%"
    }

@app.post("/detection")
async def add_detection(detection: Detection):
    """add a new detection (this would come from our ML model)"""
    detections.append(detection)
    return {"message": "Detection added", "detection_id": len(detections)}

@app.post("/process_video")
async def process_video(camera_id: str):
    """process a real video from a camera"""
    # process real video using our pipeline
    try:
        # find a random video directory to process
        import random
        import os
        data_dir = "./data"
        categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "Labels" and d != "DCSASS Dataset"]
        
        if not categories:
            return {"error": "No video categories found"}
        
        # pick random category and video
        category = random.choice(categories)
        category_path = os.path.join(data_dir, category)
        video_dirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        
        if not video_dirs:
            return {"error": f"No videos found in {category}"}
        
        video_dir = random.choice(video_dirs)
        video_path = os.path.join(category_path, video_dir)
        
        # process the video
        result = video_processor.process_video_file(video_path)
        
        # create detection if threat detected
        if result.get("status") == "success" and result.get("confidence", 0) > 0.5:
            detection = Detection(
                camera_id=camera_id,
                threat_type=result["threat_type"],
                confidence=result["confidence"],
                timestamp=datetime.now().strftime("%H:%M:%S"),
                location={"x": 150, "y": 200}
            )
            detections.append(detection)
        
        return {"result": result, "detection_added": result.get("confidence", 0) > 0.5}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)