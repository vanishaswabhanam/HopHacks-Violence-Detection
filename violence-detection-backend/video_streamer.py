import cv2
import threading
import time
import os
import json
from typing import Dict, Optional
import numpy as np

class RealVideoStreamer:
    """Real video streaming from DCSASS dataset"""
    
    def __init__(self):
        self.streams = {}
        self.data_dir = "data"
        self.video_categories = [
            "RoadAccidents", "Arson", "Shoplifting", "Stealing", "Burglary",
            "Explosion", "Fighting", "Robbery", "Shooting", "Vandalism",
            "Normal", "Abuse", "Arrest"
        ]
        
    def get_available_videos(self) -> Dict[str, list]:
        """Get available videos by category"""
        videos = {}
        for category in self.video_categories:
            category_path = os.path.join(self.data_dir, category)
            if os.path.exists(category_path):
                videos[category] = [f for f in os.listdir(category_path) if f.endswith('.mp4')]
            else:
                videos[category] = []
        return videos
    
    def start_stream(self, stream_id: str, camera_id: str = None) -> Dict:
        """Start streaming a video"""
        if stream_id in self.streams:
            return {"status": "already_running", "stream_id": stream_id}
        
        # Get available videos
        available_videos = self.get_available_videos()
        
        # Select a video based on camera_id or random
        selected_video = None
        selected_category = None
        
        if camera_id:
            # Map camera IDs to specific categories for demo
            camera_mapping = {
                "cam_7": "Fighting",
                "cam_8": "Normal", 
                "cam_9": "Robbery",
                "cam_10": "Shoplifting"
            }
            category = camera_mapping.get(camera_id, "Normal")
            if available_videos.get(category):
                selected_video = available_videos[category][0]
                selected_category = category
        else:
            # Pick any available video
            for category, videos in available_videos.items():
                if videos:
                    selected_video = videos[0]
                    selected_category = category
                    break
        
        if not selected_video:
            return {"status": "error", "message": "No videos available"}
        
        video_path = os.path.join(self.data_dir, selected_category, selected_video)
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "message": f"Could not open video: {video_path}"}
        
        # Store stream data
        self.streams[stream_id] = {
            "camera_id": camera_id,
            "video_path": video_path,
            "video_name": selected_video,
            "category": selected_category,
            "cap": cap,
            "current_frame": None,
            "frame_count": 0,
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "running": True,
            "thread": None
        }
        
        # Start streaming thread
        stream_thread = threading.Thread(target=self._stream_worker, args=(stream_id,))
        stream_thread.daemon = True
        stream_thread.start()
        self.streams[stream_id]["thread"] = stream_thread
        
        return {
            "status": "started",
            "stream_id": stream_id,
            "video_path": video_path,
            "video_name": selected_video,
            "category": selected_category,
            "fps": self.streams[stream_id]["fps"],
            "total_frames": self.streams[stream_id]["total_frames"]
        }
    
    def stop_stream(self, stream_id: str) -> Dict:
        """Stop a video stream"""
        if stream_id not in self.streams:
            return {"status": "not_found", "stream_id": stream_id}
        
        stream_data = self.streams[stream_id]
        stream_data["running"] = False
        
        if stream_data["cap"]:
            stream_data["cap"].release()
        
        if stream_data["thread"] and stream_data["thread"].is_alive():
            stream_data["thread"].join(timeout=2)
        
        del self.streams[stream_id]
        
        return {"status": "stopped", "stream_id": stream_id}
    
    def get_stream_info(self, stream_id: str) -> Dict:
        """Get information about a stream"""
        if stream_id not in self.streams:
            return {"status": "not_found"}
        
        stream_data = self.streams[stream_id]
        return {
            "status": "running" if stream_data["running"] else "stopped",
            "stream_id": stream_id,
            "camera_id": stream_data["camera_id"],
            "video_path": stream_data["video_path"],
            "video_name": stream_data["video_name"],
            "category": stream_data["category"],
            "fps": stream_data["fps"],
            "total_frames": stream_data["total_frames"],
            "current_frame": stream_data["frame_count"]
        }
    
    def get_all_streams(self) -> Dict:
        """Get information about all streams"""
        streams_info = {}
        for stream_id in self.streams:
            streams_info[stream_id] = self.get_stream_info(stream_id)
        return streams_info
    
    def get_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """Get the current frame from a stream"""
        if stream_id not in self.streams:
            return None
        
        stream_data = self.streams[stream_id]
        return stream_data["current_frame"]
    
    def _stream_worker(self, stream_id: str):
        """Worker thread for streaming video"""
        stream_data = self.streams[stream_id]
        cap = stream_data["cap"]
        
        frame_interval = 1.0 / stream_data["fps"]
        
        while stream_data["running"]:
            ret, frame = cap.read()
            if not ret:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            stream_data["current_frame"] = frame
            stream_data["frame_count"] += 1
            
            time.sleep(frame_interval)
        
        cap.release()

# Global instance
video_streamer = RealVideoStreamer()
