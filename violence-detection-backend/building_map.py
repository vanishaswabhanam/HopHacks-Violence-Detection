"""
Building Map Integration System
Shows threat locations and camera positions on building floor plan
"""

import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MapPoint:
    """Represents a point on the building map"""
    x: float
    y: float
    label: str
    point_type: str  # 'camera', 'threat', 'incident', 'zone'
    metadata: Dict = None

@dataclass
class ThreatLocation:
    """Represents a threat location on the map"""
    threat_id: str
    threat_type: str
    camera_id: str
    confidence: float
    timestamp: str
    position: Tuple[float, float]
    severity: str
    status: str  # 'active', 'resolved', 'investigating'

class BuildingMapSystem:
    """Building map system for threat visualization"""
    
    def __init__(self):
        # Building dimensions (in map units)
        self.map_width = 800
        self.map_height = 600
        
        # Camera locations (relative coordinates 0-1)
        self.camera_positions = {
            'cam_1': {'x': 0.2, 'y': 0.3, 'name': 'Hallway A', 'zone': 'North Wing'},
            'cam_2': {'x': 0.7, 'y': 0.4, 'name': 'Cafeteria', 'zone': 'Main Floor'},
            'cam_3': {'x': 0.5, 'y': 0.8, 'name': 'Main Entrance', 'zone': 'Ground Floor'},
            'cam_4': {'x': 0.8, 'y': 0.2, 'name': 'Gymnasium', 'zone': 'East Wing'}
        }
        
        # Building zones
        self.building_zones = {
            'North Wing': {'x': 0.1, 'y': 0.1, 'width': 0.4, 'height': 0.5},
            'Main Floor': {'x': 0.5, 'y': 0.2, 'width': 0.4, 'height': 0.4},
            'Ground Floor': {'x': 0.3, 'y': 0.6, 'width': 0.4, 'height': 0.3},
            'East Wing': {'x': 0.7, 'y': 0.1, 'width': 0.3, 'height': 0.3}
        }
        
        # Threat severity colors
        self.threat_colors = {
            'low': '#00ff00',      # Green
            'medium': '#ffaa00',   # Yellow  
            'high': '#ff4444',     # Red
            'critical': '#8b0000'  # Dark Red
        }
        
        # Active threats tracking
        self.active_threats = {}
        self.threat_history = []
        
        print("✅ Building map system initialized")
    
    def convert_to_map_coordinates(self, camera_id: str, detection_location: Dict) -> Tuple[float, float]:
        """Convert detection location to map coordinates"""
        try:
            # Get camera position
            camera_pos = self.camera_positions.get(camera_id)
            if not camera_pos:
                return (0.5, 0.5)  # Default center position
            
            # Convert detection coordinates to map coordinates
            # Detection coordinates are relative to camera view
            detection_x = detection_location.get('x', 150)
            detection_y = detection_location.get('y', 200)
            
            # Normalize detection coordinates (assuming 320x240 camera resolution)
            norm_x = detection_x / 320.0
            norm_y = detection_y / 240.0
            
            # Map to building coordinates with some randomness for realistic positioning
            import random
            map_x = camera_pos['x'] + (norm_x - 0.5) * 0.1 + random.uniform(-0.05, 0.05)
            map_y = camera_pos['y'] + (norm_y - 0.5) * 0.1 + random.uniform(-0.05, 0.05)
            
            # Clamp to map bounds
            map_x = max(0.05, min(0.95, map_x))
            map_y = max(0.05, min(0.95, map_y))
            
            return (map_x, map_y)
            
        except Exception as e:
            print(f"❌ Error converting coordinates: {e}")
            return (0.5, 0.5)
    
    def add_threat_location(self, detection: Dict) -> str:
        """Add threat location to map"""
        try:
            camera_id = detection.get('camera_id', 'unknown')
            threat_type = detection.get('threat_type', 'unknown')
            confidence = detection.get('confidence', 0.0)
            severity = detection.get('severity', 'low')
            location = detection.get('location', {'x': 150, 'y': 200})
            timestamp = detection.get('timestamp', datetime.now().strftime('%H:%M:%S'))
            
            # Convert to map coordinates
            map_position = self.convert_to_map_coordinates(camera_id, location)
            
            # Generate threat ID
            threat_id = f"threat_{len(self.active_threats) + 1}_{int(datetime.now().timestamp())}"
            
            # Create threat location
            threat_location = ThreatLocation(
                threat_id=threat_id,
                threat_type=threat_type,
                camera_id=camera_id,
                confidence=confidence,
                timestamp=timestamp,
                position=map_position,
                severity=severity,
                status='active'
            )
            
            # Add to active threats
            self.active_threats[threat_id] = threat_location
            
            # Add to history
            self.threat_history.append(threat_location)
            
            print(f"✅ Threat location added: {threat_type} at {map_position}")
            return threat_id
            
        except Exception as e:
            print(f"❌ Error adding threat location: {e}")
            return None
    
    def resolve_threat(self, threat_id: str):
        """Mark threat as resolved"""
        if threat_id in self.active_threats:
            self.active_threats[threat_id].status = 'resolved'
            print(f"✅ Threat resolved: {threat_id}")
    
    def get_map_data(self) -> Dict:
        """Get complete map data for visualization"""
        try:
            # Convert active threats to map points
            threat_points = []
            for threat_id, threat in self.active_threats.items():
                if threat.status == 'active':
                    threat_points.append(MapPoint(
                        x=threat.position[0] * self.map_width,
                        y=threat.position[1] * self.map_height,
                        label=f"{threat.threat_type.title()} ({threat.confidence:.0%})",
                        point_type='threat',
                        metadata={
                            'threat_id': threat_id,
                            'threat_type': threat.threat_type,
                            'camera_id': threat.camera_id,
                            'confidence': threat.confidence,
                            'severity': threat.severity,
                            'timestamp': threat.timestamp,
                            'color': self.threat_colors.get(threat.severity, '#666666')
                        }
                    ))
            
            # Convert cameras to map points
            camera_points = []
            for camera_id, camera_info in self.camera_positions.items():
                camera_points.append(MapPoint(
                    x=camera_info['x'] * self.map_width,
                    y=camera_info['y'] * self.map_height,
                    label=camera_info['name'],
                    point_type='camera',
                    metadata={
                        'camera_id': camera_id,
                        'name': camera_info['name'],
                        'zone': camera_info['zone'],
                        'status': 'online',
                        'color': '#0099ff'
                    }
                ))
            
            # Convert zones to map areas
            zone_areas = []
            for zone_name, zone_info in self.building_zones.items():
                zone_areas.append({
                    'name': zone_name,
                    'x': zone_info['x'] * self.map_width,
                    'y': zone_info['y'] * self.map_height,
                    'width': zone_info['width'] * self.map_width,
                    'height': zone_info['height'] * self.map_height,
                    'color': '#f0f0f0',
                    'border_color': '#cccccc'
                })
            
            # Recent incidents (last 10)
            recent_incidents = []
            for threat in self.threat_history[-10:]:
                recent_incidents.append({
                    'threat_id': threat.threat_id,
                    'threat_type': threat.threat_type,
                    'camera_id': threat.camera_id,
                    'confidence': threat.confidence,
                    'timestamp': threat.timestamp,
                    'severity': threat.severity,
                    'status': threat.status,
                    'position': {
                        'x': threat.position[0] * self.map_width,
                        'y': threat.position[1] * self.map_height
                    }
                })
            
            map_data = {
                'map_info': {
                    'width': self.map_width,
                    'height': self.map_height,
                    'last_updated': datetime.now().isoformat(),
                    'total_threats': len(self.active_threats),
                    'active_threats': len([t for t in self.active_threats.values() if t.status == 'active'])
                },
                'zones': zone_areas,
                'cameras': [
                    {
                        'x': point.x,
                        'y': point.y,
                        'label': point.label,
                        'metadata': point.metadata
                    } for point in camera_points
                ],
                'threats': [
                    {
                        'x': point.x,
                        'y': point.y,
                        'label': point.label,
                        'metadata': point.metadata
                    } for point in threat_points
                ],
                'recent_incidents': recent_incidents,
                'threat_summary': self.get_threat_summary()
            }
            
            return map_data
            
        except Exception as e:
            print(f"❌ Error getting map data: {e}")
            return {'error': str(e)}
    
    def get_threat_summary(self) -> Dict:
        """Get threat summary statistics"""
        active_threats = [t for t in self.active_threats.values() if t.status == 'active']
        
        summary = {
            'total_active': len(active_threats),
            'by_severity': {},
            'by_type': {},
            'by_camera': {},
            'by_zone': {}
        }
        
        for threat in active_threats:
            # By severity
            severity = threat.severity
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # By type
            threat_type = threat.threat_type
            summary['by_type'][threat_type] = summary['by_type'].get(threat_type, 0) + 1
            
            # By camera
            camera_id = threat.camera_id
            summary['by_camera'][camera_id] = summary['by_camera'].get(camera_id, 0) + 1
            
            # By zone
            camera_info = self.camera_positions.get(camera_id, {})
            zone = camera_info.get('zone', 'Unknown')
            summary['by_zone'][zone] = summary['by_zone'].get(zone, 0) + 1
        
        return summary
    
    def get_camera_coverage(self) -> Dict:
        """Get camera coverage analysis"""
        coverage = {}
        
        for camera_id, camera_info in self.camera_positions.items():
            # Count threats in this camera's area
            threats_in_area = 0
            for threat in self.active_threats.values():
                if threat.camera_id == camera_id and threat.status == 'active':
                    threats_in_area += 1
            
            coverage[camera_id] = {
                'name': camera_info['name'],
                'zone': camera_info['zone'],
                'position': {
                    'x': camera_info['x'] * self.map_width,
                    'y': camera_info['y'] * self.map_height
                },
                'active_threats': threats_in_area,
                'status': 'high_alert' if threats_in_area > 0 else 'normal'
            }
        
        return coverage
    
    def simulate_threat_movement(self, threat_id: str, new_position: Tuple[float, float]):
        """Simulate threat moving between camera zones"""
        if threat_id in self.active_threats:
            old_position = self.active_threats[threat_id].position
            self.active_threats[threat_id].position = new_position
            
            print(f"🔄 Threat {threat_id} moved from {old_position} to {new_position}")
    
    def cleanup_old_threats(self, hours_to_keep: int = 24):
        """Clean up old resolved threats"""
        cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
        removed_count = 0
        
        # Remove old resolved threats from active threats
        threats_to_remove = []
        for threat_id, threat in self.active_threats.items():
            if threat.status == 'resolved':
                try:
                    threat_time = datetime.strptime(threat.timestamp, '%H:%M:%S')
                    if threat_time < cutoff_time:
                        threats_to_remove.append(threat_id)
                except:
                    threats_to_remove.append(threat_id)
        
        for threat_id in threats_to_remove:
            del self.active_threats[threat_id]
            removed_count += 1
        
        print(f"🧹 Cleaned up {removed_count} old threats")
    
    def export_map_data(self, file_path: str):
        """Export map data to JSON file"""
        try:
            map_data = self.get_map_data()
            with open(file_path, 'w') as f:
                json.dump(map_data, f, indent=2)
            print(f"✅ Map data exported to {file_path}")
        except Exception as e:
            print(f"❌ Error exporting map data: {e}")

# Global building map instance
building_map = BuildingMapSystem()

def add_threat_to_map(detection: Dict) -> str:
    """Add threat location to building map"""
    return building_map.add_threat_location(detection)

def get_building_map_data() -> Dict:
    """Get building map data"""
    return building_map.get_map_data()

def get_threat_summary() -> Dict:
    """Get threat summary"""
    return building_map.get_threat_summary()

def get_camera_coverage() -> Dict:
    """Get camera coverage analysis"""
    return building_map.get_camera_coverage()

def resolve_threat_on_map(threat_id: str):
    """Resolve threat on map"""
    building_map.resolve_threat(threat_id)

# Test function
def test_building_map():
    """Test building map system"""
    print("🧪 Testing Building Map System...")
    
    # Test adding threats
    sample_detections = [
        {
            "camera_id": "cam_1",
            "threat_type": "fighting",
            "confidence": 0.95,
            "severity": "high",
            "location": {"x": 150, "y": 200},
            "timestamp": "17:30:00"
        },
        {
            "camera_id": "cam_2", 
            "threat_type": "robbery",
            "confidence": 0.88,
            "severity": "high",
            "location": {"x": 200, "y": 150},
            "timestamp": "17:35:00"
        }
    ]
    
    print("\n📍 Adding threat locations...")
    for detection in sample_detections:
        threat_id = add_threat_to_map(detection)
        print(f"  Added threat: {detection['threat_type']} at {detection['camera_id']}")
    
    # Test map data
    print("\n🗺️ Map data:")
    map_data = get_building_map_data()
    print(f"  Map size: {map_data['map_info']['width']}x{map_data['map_info']['height']}")
    print(f"  Active threats: {map_data['map_info']['active_threats']}")
    print(f"  Cameras: {len(map_data['cameras'])}")
    print(f"  Zones: {len(map_data['zones'])}")
    
    # Test threat summary
    print("\n📊 Threat summary:")
    summary = get_threat_summary()
    print(f"  Total active: {summary['total_active']}")
    print(f"  By severity: {summary['by_severity']}")
    print(f"  By type: {summary['by_type']}")
    
    print("\n✅ Building map test completed!")

if __name__ == "__main__":
    test_building_map()
