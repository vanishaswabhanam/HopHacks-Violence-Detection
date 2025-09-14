from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import cv2
import io
from real_video_processor import RealVideoProcessor
from redis_mcp_coordination import RedisMCPCoordinationHub
from discord_notifications import send_discord_alert_sync, send_discord_summary, setup_discord_webhook, test_discord_connection
from evidence_collection import collect_evidence_for_detection, get_evidence_summary, list_evidence_files, get_evidence_by_id
from building_map import add_threat_to_map, get_building_map_data, get_threat_summary, get_camera_coverage, resolve_threat_on_map
from video_streamer import video_streamer
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

# initialize Redis MCP coordination hub (lazy initialization)
mcp_hub = None

def get_mcp_hub():
    global mcp_hub
    if mcp_hub is None:
        mcp_hub = RedisMCPCoordinationHub()
    return mcp_hub

def find_source_video_for_camera(camera_id: str) -> Optional[str]:
    """Find source video file for a camera"""
    try:
        hub = get_mcp_hub()
        camera_system = hub.camera_system
        
        # Get camera info
        camera_info = camera_system.cameras.get(camera_id)
        if camera_info and 'current_video' in camera_info:
            video_path = camera_info['current_video']
            if os.path.exists(video_path):
                return video_path
        
        # Fallback: look for any video in the camera's directory
        camera_dir = camera_system.camera_directories.get(camera_id)
        if camera_dir and os.path.exists(camera_dir):
            for file in os.listdir(camera_dir):
                if file.endswith('.mp4'):
                    return os.path.join(camera_dir, file)
        
        return None
    except Exception as e:
        print(f"❌ Error finding source video for {camera_id}: {e}")
        return None

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
    """get system statistics for React dashboard"""
    # get camera status from MCP hub
    hub = get_mcp_hub()
    camera_status = hub.camera_system.get_camera_status()
    
    # count alerts from detections
    alerts_today = len([d for d in detections if d.get('confidence', 0) > 0.7])
    
    return {
        "cameras_online": camera_status.get("online_cameras", 4),
        "total_cameras": camera_status.get("total_cameras", 4),
        "alerts_today": alerts_today,
        "system_uptime": 99.8
    }

@app.post("/detection")
async def add_detection(detection: Detection):
    """add a new detection (this would come from our ML model)"""
    detections.append(detection)
    
    detection_dict = detection.dict()
    
    # Collect evidence for high-confidence detections
    evidence_metadata = None
    threat_id = None
    if detection.confidence > 0.7:
        # Try to find source video for this camera
        source_video_path = find_source_video_for_camera(detection.camera_id)
        evidence_metadata = collect_evidence_for_detection(detection_dict, source_video_path)
        
        # Add threat to building map
        threat_id = add_threat_to_map(detection_dict)
        
        # Send Discord notification
        send_discord_alert_sync(detection_dict)
    
    return {
        "message": "Detection added", 
        "detection_id": len(detections),
        "evidence_collected": evidence_metadata is not None,
        "evidence_id": evidence_metadata.get("evidence_id") if evidence_metadata else None,
        "threat_added_to_map": threat_id is not None,
        "threat_id": threat_id
    }

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

# Multi-Camera Coordination Endpoints
@app.get("/cameras")
async def get_cameras():
    """Get all camera information"""
    hub = get_mcp_hub()
    return hub.camera_system.get_camera_status()

@app.get("/coordination/status")
async def get_coordination_status():
    """Get Redis MCP coordination status"""
    try:
        hub = get_mcp_hub()
        await hub.connect_redis()
        status = await hub.get_coordination_status()
        await hub.disconnect_redis()
        return status
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/coordination/incidents")
async def get_active_incidents():
    """Get active incidents across all cameras from Redis"""
    try:
        hub = get_mcp_hub()
        await hub.connect_redis()
        incidents = await hub.get_active_incidents_summary()
        await hub.disconnect_redis()
        return incidents
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/coordination/run")
async def run_coordination():
    """Run a Redis coordination cycle across all cameras"""
    try:
        hub = get_mcp_hub()
        await hub.connect_redis()
        results = await hub.coordinate_camera_feeds()
        await hub.disconnect_redis()
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/cameras/{camera_id}/feed")
async def get_camera_feed(camera_id: str):
    """Get feed from a specific camera"""
    try:
        hub = get_mcp_hub()
        result = hub.camera_system.process_camera_feed(camera_id)
        return {"status": "success", "feed": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Discord Integration Endpoints
@app.post("/discord/setup")
async def setup_discord(webhook_url: str):
    """Setup Discord webhook URL"""
    try:
        success = setup_discord_webhook(webhook_url)
        return {"status": "success" if success else "error", "message": "Discord webhook configured" if success else "Failed to configure webhook"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/discord/test")
async def test_discord():
    """Test Discord webhook connection"""
    try:
        success = test_discord_connection()
        return {"status": "success" if success else "error", "message": "Discord webhook test completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/discord/send-summary")
async def send_discord_summary_endpoint():
    """Send system summary to Discord"""
    try:
        # Get current stats
        camera_status = get_mcp_hub().camera_system.get_camera_status()
        alerts_today = len([d for d in detections if d.get('confidence', 0) > 0.7])
        
        stats = {
            "cameras_online": camera_status.get("online_cameras", 4),
            "total_cameras": camera_status.get("total_cameras", 4),
            "alerts_today": alerts_today,
            "system_uptime": 99.8
        }
        
        success = await send_discord_summary(stats)
        return {"status": "success" if success else "error", "message": "Discord summary sent" if success else "Failed to send summary"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Evidence Collection Endpoints
@app.get("/evidence/summary")
async def get_evidence_summary_endpoint():
    """Get evidence collection summary"""
    try:
        summary = get_evidence_summary()
        return {"status": "success", "summary": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/evidence/files")
async def list_evidence_files_endpoint():
    """List all evidence files"""
    try:
        files = list_evidence_files()
        return {"status": "success", "files": files}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/evidence/{evidence_id}")
async def get_evidence_endpoint(evidence_id: str):
    """Get specific evidence by ID"""
    try:
        evidence = get_evidence_by_id(evidence_id)
        if evidence:
            return {"status": "success", "evidence": evidence}
        else:
            return {"status": "error", "message": "Evidence not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/evidence/collect")
async def collect_evidence_endpoint(detection: Detection):
    """Manually collect evidence for a detection"""
    try:
        detection_dict = detection.dict()
        source_video_path = find_source_video_for_camera(detection.camera_id)
        evidence_metadata = collect_evidence_for_detection(detection_dict, source_video_path)
        
        return {
            "status": "success", 
            "message": "Evidence collected",
            "evidence_id": evidence_metadata.get("evidence_id"),
            "video_clip_path": evidence_metadata.get("evidence", {}).get("video_clip_path")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Building Map Endpoints
@app.get("/map/data")
async def get_map_data_endpoint():
    """Get building map data with threat locations"""
    try:
        map_data = get_building_map_data()
        return {"status": "success", "map_data": map_data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/map/threats")
async def get_map_threats_endpoint():
    """Get threat summary for map"""
    try:
        threat_summary = get_threat_summary()
        return {"status": "success", "threat_summary": threat_summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/map/coverage")
async def get_map_coverage_endpoint():
    """Get camera coverage analysis"""
    try:
        coverage = get_camera_coverage()
        return {"status": "success", "coverage": coverage}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/map/resolve-threat/{threat_id}")
async def resolve_threat_endpoint(threat_id: str):
    """Resolve a threat on the map"""
    try:
        resolve_threat_on_map(threat_id)
        return {"status": "success", "message": f"Threat {threat_id} resolved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Video Streaming Endpoints
@app.get("/video/categories")
async def get_video_categories():
    """Get available video categories from DCSASS dataset"""
    try:
        categories = video_streamer.get_available_categories()
        return {"status": "success", "categories": categories}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/video/category/{category}")
async def get_videos_by_category(category: str):
    """Get videos from a specific category"""
    try:
        videos = video_streamer.get_videos_by_category(category)
        return {"status": "success", "category": category, "videos": videos}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/video/stream/start/{stream_id}")
async def start_video_stream(stream_id: str, video_path: Optional[str] = None, category: Optional[str] = None):
    """Start streaming a video"""
    try:
        success = video_streamer.start_stream(stream_id, video_path, category)
        if success:
            return {"status": "success", "message": f"Stream {stream_id} started"}
        else:
            return {"status": "error", "message": f"Failed to start stream {stream_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/video/stream/stop/{stream_id}")
async def stop_video_stream(stream_id: str):
    """Stop a video stream"""
    try:
        success = video_streamer.stop_stream(stream_id)
        if success:
            return {"status": "success", "message": f"Stream {stream_id} stopped"}
        else:
            return {"status": "error", "message": f"Stream {stream_id} not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/video/stream/{stream_id}")
async def get_stream_info(stream_id: str):
    """Get information about a video stream"""
    try:
        info = video_streamer.get_stream_info(stream_id)
        if info:
            return {"status": "success", "stream_info": info}
        else:
            return {"status": "error", "message": f"Stream {stream_id} not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/video/streams")
async def get_all_streams():
    """Get information about all active streams"""
    try:
        streams = video_streamer.get_all_streams()
        return {"status": "success", "streams": streams}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/video/frame/{stream_id}")
async def get_video_frame(stream_id: str):
    """Get current video frame from a stream"""
    try:
        frame = video_streamer.get_frame(stream_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="Stream not found or no frame available")
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        return StreamingResponse(
            io.BytesIO(frame_bytes),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
