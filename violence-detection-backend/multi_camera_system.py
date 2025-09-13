"""
Multi-Camera System
Simulates 4 different camera feeds with real video processing and MCP coordination
"""

import os
import random
import time
import json
from typing import Dict, List, Any
from pathlib import Path
from real_video_processor import RealVideoProcessor

class MultiCameraSystem:
    def __init__(self):
        self.cameras = {
            'cam_1': {'name': 'Hallway A', 'location': 'North Wing', 'type': 'security'},
            'cam_2': {'name': 'Cafeteria', 'location': 'Main Floor', 'type': 'public'},
            'cam_3': {'name': 'Main Entrance', 'location': 'Ground Floor', 'type': 'security'},
            'cam_4': {'name': 'Gymnasium', 'location': 'East Wing', 'type': 'recreation'}
        }
        
        self.camera_videos = {}
        self.video_processor = RealVideoProcessor()
        self.load_camera_videos()
        
        # MCP coordination state
        self.active_incidents = {}
        self.camera_states = {}
        
    def load_camera_videos(self):
        """Load different video categories for each camera"""
        data_dir = Path('data')
        if not data_dir.exists():
            print("Data directory not found")
            return
            
        # Assign different threat categories to each camera for variety
        camera_assignments = {
            'cam_1': ['Fighting', 'Assault', 'Abuse'],  # Hallway - violent incidents
            'cam_2': ['Robbery', 'Shoplifting', 'Stealing'],  # Cafeteria - theft incidents  
            'cam_3': ['Arrest', 'Abuse', 'Explosion'],  # Entrance - security incidents
            'cam_4': ['Vandalism', 'Stealing', 'Fighting']  # Gym - property damage + fights
        }
        
        for camera_id, categories in camera_assignments.items():
            self.camera_videos[camera_id] = []
            
            for category in categories:
                category_path = data_dir / category
                if category_path.exists():
                    # Get all video directories in this category
                    videos = [d for d in category_path.iterdir() 
                             if d.is_dir()]
                    
                    # Add some videos from this category
                    for video in videos[:2]:  # Take first 2 videos from each category
                        self.camera_videos[camera_id].append({
                            'path': str(video),
                            'category': category,
                            'name': video.name
                        })
        
        print(f"Loaded videos for {len(self.camera_videos)} cameras")
        for cam_id, videos in self.camera_videos.items():
            print(f"  {cam_id}: {len(videos)} videos")
    
    def get_camera_info(self, camera_id: str) -> Dict[str, Any]:
        """Get camera information"""
        if camera_id not in self.cameras:
            return {}
            
        return {
            'camera_id': camera_id,
            'name': self.cameras[camera_id]['name'],
            'location': self.cameras[camera_id]['location'],
            'type': self.cameras[camera_id]['type'],
            'status': 'online',
            'available_videos': len(self.camera_videos.get(camera_id, []))
        }
    
    def get_random_video_for_camera(self, camera_id: str) -> Dict[str, Any]:
        """Get a random video for a specific camera"""
        if camera_id not in self.camera_videos or not self.camera_videos[camera_id]:
            return {}
            
        video = random.choice(self.camera_videos[camera_id])
        return {
            'camera_id': camera_id,
            'video_path': video['path'],
            'category': video['category'],
            'video_name': video['name'],
            'timestamp': time.time()
        }
    
    def process_camera_feed(self, camera_id: str) -> Dict[str, Any]:
        """Process a single camera feed and return analysis results"""
        video_info = self.get_random_video_for_camera(camera_id)
        if not video_info:
            return {
                'camera_id': camera_id,
                'status': 'error',
                'message': 'No video available'
            }
        
        # Process the video
        result = self.video_processor.process_video_file(video_info['video_path'])
        
        # Add camera context
        camera_info = self.get_camera_info(camera_id)
        
        # Combine results
        feed_result = {
            'camera_id': camera_id,
            'camera_name': camera_info['name'],
            'camera_location': camera_info['location'],
            'camera_type': camera_info['type'],
            'video_info': video_info,
            'analysis': result,
            'timestamp': time.time()
        }
        
        # Update camera state
        self.camera_states[camera_id] = feed_result
        
        return feed_result
    
    def process_all_cameras(self) -> Dict[str, Any]:
        """Process all camera feeds simultaneously"""
        results = {}
        
        for camera_id in self.cameras.keys():
            print(f"Processing {camera_id} ({self.cameras[camera_id]['name']})...")
            results[camera_id] = self.process_camera_feed(camera_id)
        
        return results
    
    def get_camera_status(self) -> Dict[str, Any]:
        """Get status of all cameras"""
        status = {
            'total_cameras': len(self.cameras),
            'online_cameras': len(self.cameras),
            'cameras': {}
        }
        
        for camera_id in self.cameras.keys():
            status['cameras'][camera_id] = self.get_camera_info(camera_id)
            
        return status
    
    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get list of active incidents across all cameras"""
        incidents = []
        
        for camera_id, state in self.camera_states.items():
            if state.get('analysis', {}).get('status') == 'success':
                analysis = state['analysis']
                if analysis.get('threat_level') in ['high', 'critical']:
                    incidents.append({
                        'camera_id': camera_id,
                        'camera_name': state['camera_name'],
                        'camera_location': state['camera_location'],
                        'threat_level': analysis['threat_level'],
                        'threat_type': analysis.get('threat_type', 'unknown'),
                        'confidence': analysis['confidence'],
                        'timestamp': state['timestamp'],
                        'multi_class': analysis.get('multi_class_analysis', {})
                    })
        
        return incidents

def test_multi_camera_system():
    """Test the multi-camera system"""
    print("Testing Multi-Camera System...")
    
    system = MultiCameraSystem()
    
    # Test camera info
    print("\nCamera Information:")
    for camera_id in system.cameras.keys():
        info = system.get_camera_info(camera_id)
        print(f"  {camera_id}: {info['name']} ({info['location']}) - {info['type']}")
    
    # Test single camera processing
    print("\nTesting Single Camera Processing:")
    test_camera = 'cam_1'
    result = system.process_camera_feed(test_camera)
    print(f"  {test_camera}: {result['camera_name']}")
    if result['analysis']['status'] == 'success':
        analysis = result['analysis']
        print(f"    Threat Level: {analysis['threat_level']}")
        print(f"    Confidence: {analysis['confidence']}")
        if 'multi_class_analysis' in analysis:
            mc = analysis['multi_class_analysis']
            print(f"    Multi-Class: {mc['threat_type']} ({mc['confidence']:.2f}, {mc['color_code']})")
    
    # Test all cameras
    print("\nTesting All Cameras:")
    all_results = system.process_all_cameras()
    
    print("\nActive Incidents:")
    incidents = system.get_active_incidents()
    for incident in incidents:
        print(f"  {incident['camera_name']}: {incident['threat_type']} ({incident['threat_level']}, {incident['confidence']:.2f})")
    
    print("\nMulti-camera system test completed!")

if __name__ == "__main__":
    test_multi_camera_system()