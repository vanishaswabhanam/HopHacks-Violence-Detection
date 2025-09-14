"""
Video Streaming System
Simulates real-time camera feeds from DCSASS dataset
"""

import cv2
import numpy as np
import os
import random
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

class VideoStream:
    """Individual video stream for a camera"""
    
    def __init__(self, camera_id: str, video_path: str, stream_name: str):
        self.camera_id = camera_id
        self.video_path = video_path
        self.stream_name = stream_name
        self.cap = None
        self.current_frame = None
        self.frame_count = 0
        self.fps = 30
        self.is_streaming = False
        self.last_detection = None
        self.threat_overlay = None
        
        # Stream settings
        self.loop_video = True
        self.add_timestamp = True
        self.add_camera_label = True
        
        print(f"📹 Video stream initialized: {stream_name} ({camera_id})")
    
    def start_stream(self):
        """Start the video stream"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"❌ Could not open video: {self.video_path}")
                return False
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.is_streaming = True
            print(f"✅ Stream started: {self.stream_name} ({self.fps} FPS, {self.frame_count} frames)")
            return True
            
        except Exception as e:
            print(f"❌ Error starting stream: {e}")
            return False
    
    def get_next_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the stream"""
        if not self.is_streaming or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.loop_video:
                    # Restart video from beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
                else:
                    return None
            
            # Add overlays
            frame = self.add_overlays(frame)
            
            self.current_frame = frame
            return frame
            
        except Exception as e:
            print(f"❌ Error getting frame: {e}")
            return None
    
    def add_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Add overlays to the frame"""
        try:
            # Convert to PIL for easier text rendering
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Add timestamp
            if self.add_timestamp:
                current_time = datetime.now().strftime("%H:%M:%S")
                draw.text((10, 10), current_time, fill=(0, 255, 0), font_size=20)
            
            # Add camera label
            if self.add_camera_label:
                draw.text((10, frame.shape[0] - 30), self.stream_name, fill=(255, 255, 255), font_size=16)
            
            # Add threat overlay if active
            if self.threat_overlay:
                self.add_threat_overlay(draw, frame.shape)
            
            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return frame
            
        except Exception as e:
            print(f"❌ Error adding overlays: {e}")
            return frame
    
    def add_threat_overlay(self, draw: ImageDraw.Draw, frame_shape: Tuple[int, int, int]):
        """Add threat detection overlay"""
        if not self.last_detection:
            return
        
        threat_type = self.last_detection.get('threat_type', 'unknown')
        confidence = self.last_detection.get('confidence', 0.0)
        severity = self.last_detection.get('severity', 'low')
        
        # Threat colors
        colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255), # Yellow
            'high': (0, 0, 255),     # Red
            'critical': (128, 0, 128) # Purple
        }
        
        color = colors.get(severity, (255, 255, 255))
        
        # Draw threat box
        box_x, box_y = 50, 50
        box_width, box_height = 300, 80
        
        # Draw semi-transparent background
        overlay = Image.new('RGBA', (frame_shape[1], frame_shape[0]), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], 
                              fill=(0, 0, 0, 128))
        
        # Draw threat text
        threat_text = f"{threat_type.upper()} DETECTED"
        confidence_text = f"Confidence: {confidence:.0%}"
        
        draw.text((box_x + 10, box_y + 10), threat_text, fill=color, font_size=18)
        draw.text((box_x + 10, box_y + 35), confidence_text, fill=color, font_size=14)
        
        # Draw indicators
        indicators = self.last_detection.get('indicators', [])
        if indicators:
            indicator_text = indicators[0] if indicators else "Threat detected"
            draw.text((box_x + 10, box_y + 55), indicator_text, fill=color, font_size=12)
    
    def update_threat_detection(self, detection: Dict):
        """Update threat detection overlay"""
        self.last_detection = detection
        self.threat_overlay = True
        
        # Auto-clear threat overlay after 5 seconds
        def clear_threat():
            time.sleep(5)
            self.threat_overlay = False
            self.last_detection = None
        
        threading.Thread(target=clear_threat, daemon=True).start()
    
    def stop_stream(self):
        """Stop the video stream"""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        print(f"🛑 Stream stopped: {self.stream_name}")
    
    def get_frame_as_base64(self) -> Optional[str]:
        """Get current frame as base64 encoded string"""
        frame = self.get_next_frame()
        if frame is None:
            return None
        
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        except Exception as e:
            print(f"❌ Error encoding frame: {e}")
            return None

class VideoStreamingSystem:
    """Main video streaming system"""
    
    def __init__(self):
        self.streams = {}
        self.stream_threads = {}
        self.is_running = False
        
        # Camera configuration
        self.camera_configs = {
            'cam_1': {'name': 'Hallway A', 'zone': 'North Wing'},
            'cam_2': {'name': 'Cafeteria', 'zone': 'Main Floor'},
            'cam_3': {'name': 'Main Entrance', 'zone': 'Ground Floor'},
            'cam_4': {'name': 'Gymnasium', 'zone': 'East Wing'}
        }
        
        print("✅ Video streaming system initialized")
    
    def setup_camera_streams(self, dataset_path: str = "data"):
        """Setup camera streams from DCSASS dataset"""
        try:
            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                print(f"❌ Dataset directory not found: {dataset_path}")
                return False
            
            # Find video files for each camera
            camera_videos = self.find_camera_videos(dataset_dir)
            
            for camera_id, video_path in camera_videos.items():
                if video_path:
                    stream_name = self.camera_configs[camera_id]['name']
                    stream = VideoStream(camera_id, str(video_path), stream_name)
                    
                    if stream.start_stream():
                        self.streams[camera_id] = stream
                        print(f"✅ Camera stream setup: {camera_id} -> {stream_name}")
                    else:
                        print(f"❌ Failed to setup stream: {camera_id}")
            
            print(f"📹 Setup {len(self.streams)} camera streams")
            return len(self.streams) > 0
            
        except Exception as e:
            print(f"❌ Error setting up camera streams: {e}")
            return False
    
    def find_camera_videos(self, dataset_dir: Path) -> Dict[str, Optional[str]]:
        """Find video files for each camera"""
        camera_videos = {}
        
        # Map cameras to different video categories
        camera_mapping = {
            'cam_1': ['Fighting', 'Abuse'],      # Hallway A - violence
            'cam_2': ['Robbery', 'Burglary'],   # Cafeteria - theft
            'cam_3': ['Arrest', 'Assault'],     # Main Entrance - security
            'cam_4': ['Stealing', 'Shoplifting'] # Gymnasium - minor crimes
        }
        
        for camera_id, categories in camera_mapping.items():
            video_path = None
            
            # Try to find a video in the specified categories
            for category in categories:
                category_dir = dataset_dir / category
                if category_dir.exists():
                    # Find first video file
                    for video_file in category_dir.iterdir():
                        if video_file.is_file() and video_file.suffix.lower() in ['.mp4', '.avi', '.mov']:
                            video_path = str(video_file)
                            break
                    if video_path:
                        break
            
            camera_videos[camera_id] = video_path
        
        return camera_videos
    
    def start_all_streams(self):
        """Start all camera streams"""
        if self.is_running:
            return
        
        self.is_running = True
        
        for camera_id, stream in self.streams.items():
            thread = threading.Thread(target=self._stream_worker, args=(camera_id,), daemon=True)
            thread.start()
            self.stream_threads[camera_id] = thread
        
        print(f"🚀 Started {len(self.streams)} video streams")
    
    def _stream_worker(self, camera_id: str):
        """Worker thread for video streaming"""
        stream = self.streams.get(camera_id)
        if not stream:
            return
        
        frame_interval = 1.0 / stream.fps
        
        while self.is_running and stream.is_streaming:
            start_time = time.time()
            
            # Get next frame
            frame = stream.get_next_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)
    
    def stop_all_streams(self):
        """Stop all camera streams"""
        self.is_running = False
        
        for stream in self.streams.values():
            stream.stop_stream()
        
        # Wait for threads to finish
        for thread in self.stream_threads.values():
            thread.join(timeout=2)
        
        self.stream_threads.clear()
        print("🛑 All video streams stopped")
    
    def get_camera_frame(self, camera_id: str) -> Optional[str]:
        """Get current frame from camera as base64"""
        stream = self.streams.get(camera_id)
        if not stream:
            return None
        
        return stream.get_frame_as_base64()
    
    def get_all_camera_frames(self) -> Dict[str, Optional[str]]:
        """Get current frames from all cameras"""
        frames = {}
        for camera_id in self.streams.keys():
            frames[camera_id] = self.get_camera_frame(camera_id)
        return frames
    
    def update_camera_threat(self, camera_id: str, detection: Dict):
        """Update threat detection for a camera"""
        stream = self.streams.get(camera_id)
        if stream:
            stream.update_threat_detection(detection)
    
    def get_stream_status(self) -> Dict:
        """Get status of all streams"""
        status = {
            'total_streams': len(self.streams),
            'active_streams': len([s for s in self.streams.values() if s.is_streaming]),
            'is_running': self.is_running,
            'cameras': {}
        }
        
        for camera_id, stream in self.streams.items():
            status['cameras'][camera_id] = {
                'name': stream.stream_name,
                'is_streaming': stream.is_streaming,
                'fps': stream.fps,
                'has_threat': stream.threat_overlay is not None
            }
        
        return status

# Global video streaming system
video_streaming = VideoStreamingSystem()

def setup_video_streams(dataset_path: str = "data") -> bool:
    """Setup video streaming system"""
    return video_streaming.setup_camera_streams(dataset_path)

def start_video_streams():
    """Start all video streams"""
    video_streaming.start_all_streams()

def stop_video_streams():
    """Stop all video streams"""
    video_streaming.stop_all_streams()

def get_camera_frame(camera_id: str) -> Optional[str]:
    """Get camera frame as base64"""
    return video_streaming.get_camera_frame(camera_id)

def get_all_camera_frames() -> Dict[str, Optional[str]]:
    """Get all camera frames"""
    return video_streaming.get_all_camera_frames()

def update_camera_threat(camera_id: str, detection: Dict):
    """Update camera threat detection"""
    video_streaming.update_camera_threat(camera_id, detection)

def get_stream_status() -> Dict:
    """Get streaming status"""
    return video_streaming.get_stream_status()

# Test function
def test_video_streaming():
    """Test video streaming system"""
    print("🧪 Testing Video Streaming System...")
    
    # Setup streams
    print("\n📹 Setting up camera streams...")
    if not setup_video_streams():
        print("❌ Failed to setup video streams")
        return
    
    # Start streams
    print("\n🚀 Starting video streams...")
    start_video_streams()
    
    # Test frame capture
    print("\n📸 Testing frame capture...")
    for camera_id in ['cam_1', 'cam_2', 'cam_3', 'cam_4']:
        frame = get_camera_frame(camera_id)
        if frame:
            print(f"  ✅ {camera_id}: Frame captured ({len(frame)} chars)")
        else:
            print(f"  ❌ {camera_id}: No frame")
    
    # Test status
    print("\n📊 Stream status:")
    status = get_stream_status()
    print(f"  Total streams: {status['total_streams']}")
    print(f"  Active streams: {status['active_streams']}")
    print(f"  System running: {status['is_running']}")
    
    # Stop streams
    print("\n🛑 Stopping streams...")
    stop_video_streams()
    
    print("\n✅ Video streaming test completed!")

if __name__ == "__main__":
    test_video_streaming()
