#!/usr/bin/env python3
"""
Real Video Streaming System using DCSASS Dataset
Streams real videos from the dataset and applies violence detection
"""

import cv2
import os
import random
import threading
import time
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

class RealVideoStreamer:
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.video_files = self._discover_videos()
        self.active_streams: Dict[str, Dict] = {}
        self.stream_threads: Dict[str, threading.Thread] = {}
        
    def _discover_videos(self) -> List[str]:
        """Discover all video files in the dataset"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for category_dir in self.data_path.iterdir():
            if category_dir.is_dir() and category_dir.name not in ['Labels', 'multi_class_training_frames', 'training_frames']:
                for video_file in category_dir.rglob('*'):
                    if video_file.suffix.lower() in video_extensions:
                        video_files.append(str(video_file))
        
        print(f"Found {len(video_files)} video files")
        return video_files
    
    def get_available_categories(self) -> List[str]:
        """Get list of available video categories"""
        categories = []
        for category_dir in self.data_path.iterdir():
            if category_dir.is_dir() and category_dir.name not in ['Labels', 'multi_class_training_frames', 'training_frames']:
                categories.append(category_dir.name)
        return categories
    
    def get_videos_by_category(self, category: str) -> List[str]:
        """Get all videos from a specific category"""
        category_path = self.data_path / category
        if not category_path.exists():
            return []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        
        for video_file in category_path.rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                videos.append(str(video_file))
        
        return videos
    
    def start_stream(self, stream_id: str, video_path: Optional[str] = None, category: Optional[str] = None) -> bool:
        """Start streaming a video"""
        if stream_id in self.active_streams:
            print(f"Stream {stream_id} already active")
            return False
        
        # Select video
        if video_path:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return False
            selected_video = video_path
        elif category:
            category_videos = self.get_videos_by_category(category)
            if not category_videos:
                print(f"No videos found in category: {category}")
                return False
            selected_video = random.choice(category_videos)
        else:
            selected_video = random.choice(self.video_files)
        
        print(f"Starting stream {stream_id} with video: {selected_video}")
        
        # Initialize stream data
        self.active_streams[stream_id] = {
            'video_path': selected_video,
            'cap': None,
            'frame_count': 0,
            'fps': 30,
            'width': 640,
            'height': 480,
            'violence_detected': False,
            'last_detection': None,
            'running': True
        }
        
        # Start streaming thread
        thread = threading.Thread(target=self._stream_worker, args=(stream_id,))
        thread.daemon = True
        thread.start()
        self.stream_threads[stream_id] = thread
        
        return True
    
    def _stream_worker(self, stream_id: str):
        """Worker thread for streaming video"""
        stream_data = self.active_streams[stream_id]
        video_path = stream_data['video_path']
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            del self.active_streams[stream_id]
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        stream_data.update({
            'cap': cap,
            'fps': fps,
            'width': width,
            'height': height
        })
        
        frame_delay = 1.0 / fps
        
        while stream_data['running']:
            ret, frame = cap.read()
            if not ret:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Store current frame for display
            stream_data['current_frame'] = frame
            
            # Simulate violence detection (replace with real model)
            violence_detected = self._simulate_violence_detection(frame, stream_id)
            
            # Update stream data
            stream_data.update({
                'frame_count': stream_data['frame_count'] + 1,
                'violence_detected': violence_detected,
                'last_detection': time.time() if violence_detected else stream_data['last_detection']
            })
            
            time.sleep(frame_delay)
        
        cap.release()
        print(f"Stream {stream_id} stopped")
    
    def _simulate_violence_detection(self, frame, stream_id: str) -> bool:
        """Simulate violence detection (replace with real model)"""
        # Random detection for demo purposes
        # In real implementation, this would use the trained model
        return random.random() < 0.05  # 5% chance of detection
    
    def stop_stream(self, stream_id: str) -> bool:
        """Stop a video stream"""
        if stream_id not in self.active_streams:
            return False
        
        self.active_streams[stream_id]['running'] = False
        
        if stream_id in self.stream_threads:
            self.stream_threads[stream_id].join(timeout=5)
            del self.stream_threads[stream_id]
        
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        
        print(f"Stopped stream {stream_id}")
        return True
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict]:
        """Get information about a stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream_data = self.active_streams[stream_id].copy()
        # Remove cap object for JSON serialization
        if 'cap' in stream_data:
            del stream_data['cap']
        
        return stream_data
    
    def get_all_streams(self) -> Dict[str, Dict]:
        """Get information about all active streams"""
        streams = {}
        for stream_id in self.active_streams:
            streams[stream_id] = self.get_stream_info(stream_id)
        return streams
    
    def get_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """Get current frame from a stream (for display)"""
        if stream_id not in self.active_streams:
            return None
        
        stream_data = self.active_streams[stream_id]
        cap = stream_data.get('cap')
        
        if cap is None:
            return None
        
        # Get current frame position
        current_frame = stream_data.get('current_frame', None)
        if current_frame is not None:
            return current_frame
        
        # If no current frame, read a new one
        ret, frame = cap.read()
        if ret:
            # Store the current frame for reuse
            stream_data['current_frame'] = frame
            return frame
        
        return None

# Global streamer instance
video_streamer = RealVideoStreamer()

if __name__ == "__main__":
    # Test the video streamer
    print("Testing Real Video Streamer")
    
    # Get available categories
    categories = video_streamer.get_available_categories()
    print(f"Available categories: {categories}")
    
    # Start a test stream
    test_stream_id = "test_stream_1"
    if video_streamer.start_stream(test_stream_id, category="Fighting"):
        print(f"Started test stream: {test_stream_id}")
        
        # Let it run for a bit
        time.sleep(10)
        
        # Get stream info
        info = video_streamer.get_stream_info(test_stream_id)
        print(f"Stream info: {info}")
        
        # Stop the stream
        video_streamer.stop_stream(test_stream_id)
        print("Test completed")
    else:
        print("Failed to start test stream")
