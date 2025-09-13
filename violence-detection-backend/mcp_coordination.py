"""
MCP Coordination Hub
Manages message passing and coordination between multiple camera agents
"""

import time
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from multi_camera_system import MultiCameraSystem

@dataclass
class DetectionMessage:
    """Message format for camera detections"""
    message_id: str
    camera_id: str
    incident_id: str
    threat_type: str
    threat_level: str
    confidence: float
    location: Dict[str, Any]
    timestamp: float
    indicators: List[str]
    multi_class_analysis: Optional[Dict[str, Any]] = None

@dataclass
class IncidentContext:
    """Context for tracking incidents across cameras"""
    incident_id: str
    primary_camera: str
    threat_type: str
    threat_level: str
    start_time: float
    cameras_involved: List[str]
    status: str  # active, resolved, escalated
    evidence: List[Dict[str, Any]]

class MCPCoordinationHub:
    """Multi-Camera Protocol Coordination Hub"""
    
    def __init__(self):
        self.camera_system = MultiCameraSystem()
        self.active_incidents = {}
        self.message_queue = []
        self.camera_states = {}
        self.coordination_rules = {
            'duplicate_threshold': 0.8,  # confidence threshold for duplicate detection
            'escalation_threshold': 0.9,  # confidence threshold for escalation
            'time_window': 30,  # seconds to consider detections as related
            'max_cameras_per_incident': 4
        }
        
    def generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        return f"inc_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def create_detection_message(self, camera_id: str, analysis_result: Dict[str, Any]) -> DetectionMessage:
        """Create a detection message from camera analysis"""
        incident_id = self.generate_incident_id()
        
        return DetectionMessage(
            message_id=f"msg_{int(time.time())}_{str(uuid.uuid4())[:8]}",
            camera_id=camera_id,
            incident_id=incident_id,
            threat_type=analysis_result.get('threat_type', 'unknown'),
            threat_level=analysis_result.get('threat_level', 'low'),
            confidence=analysis_result.get('confidence', 0.0),
            location={'x': 150, 'y': 200},  # simulated location
            timestamp=time.time(),
            indicators=analysis_result.get('indicators', []),
            multi_class_analysis=analysis_result.get('multi_class_analysis')
        )
    
    def process_detection_message(self, message: DetectionMessage) -> Dict[str, Any]:
        """Process a detection message and determine coordination actions"""
        actions = []
        
        # Check for duplicate incidents
        duplicate_incident = self.check_for_duplicate_incident(message)
        if duplicate_incident:
            actions.append({
                'type': 'merge_incident',
                'target_incident': duplicate_incident,
                'reason': 'Duplicate detection across cameras'
            })
            message.incident_id = duplicate_incident
        
        # Create or update incident context
        if message.incident_id not in self.active_incidents:
            self.active_incidents[message.incident_id] = IncidentContext(
                incident_id=message.incident_id,
                primary_camera=message.camera_id,
                threat_type=message.threat_type,
                threat_level=message.threat_level,
                start_time=message.timestamp,
                cameras_involved=[message.camera_id],
                status='active',
                evidence=[asdict(message)]
            )
        else:
            # Update existing incident
            incident = self.active_incidents[message.incident_id]
            if message.camera_id not in incident.cameras_involved:
                incident.cameras_involved.append(message.camera_id)
            incident.evidence.append(asdict(message))
            
            # Check for escalation
            if message.confidence > self.coordination_rules['escalation_threshold']:
                actions.append({
                    'type': 'escalate_incident',
                    'reason': f'High confidence detection: {message.confidence:.2f}'
                })
                incident.status = 'escalated'
        
        # Update camera state
        self.camera_states[message.camera_id] = {
            'last_detection': message.timestamp,
            'current_incident': message.incident_id,
            'threat_level': message.threat_level
        }
        
        # Add to message queue
        self.message_queue.append(asdict(message))
        
        return {
            'message_id': message.message_id,
            'incident_id': message.incident_id,
            'actions': actions,
            'status': 'processed'
        }
    
    def check_for_duplicate_incident(self, message: DetectionMessage) -> Optional[str]:
        """Check if this detection matches an existing incident"""
        current_time = message.timestamp
        
        for incident_id, incident in self.active_incidents.items():
            # Check time window
            if current_time - incident.start_time > self.coordination_rules['time_window']:
                continue
                
            # Check threat type similarity
            if incident.threat_type == message.threat_type:
                # Check if cameras are adjacent (simplified logic)
                if self.are_cameras_adjacent(incident.cameras_involved[-1], message.camera_id):
                    return incident_id
        
        return None
    
    def are_cameras_adjacent(self, camera1: str, camera2: str) -> bool:
        """Check if two cameras are adjacent (simplified logic)"""
        # Simple adjacency rules based on camera IDs
        adjacency_map = {
            'cam_1': ['cam_2', 'cam_3'],  # Hallway A connects to Cafeteria and Main Entrance
            'cam_2': ['cam_1', 'cam_4'],  # Cafeteria connects to Hallway A and Gymnasium
            'cam_3': ['cam_1', 'cam_4'],  # Main Entrance connects to Hallway A and Gymnasium
            'cam_4': ['cam_2', 'cam_3']   # Gymnasium connects to Cafeteria and Main Entrance
        }
        
        return camera2 in adjacency_map.get(camera1, [])
    
    def coordinate_camera_feeds(self) -> Dict[str, Any]:
        """Coordinate all camera feeds and process detections"""
        print("Starting MCP Coordination...")
        
        # Process all camera feeds
        camera_results = self.camera_system.process_all_cameras()
        
        coordination_results = {
            'timestamp': time.time(),
            'cameras_processed': len(camera_results),
            'messages_sent': 0,
            'incidents_created': 0,
            'incidents_updated': 0,
            'actions_taken': []
        }
        
        # Process each camera's results
        for camera_id, result in camera_results.items():
            if result.get('analysis', {}).get('status') == 'success':
                analysis = result['analysis']
                
                # Only process high-confidence detections
                if analysis.get('confidence', 0) > 0.5:
                    message = self.create_detection_message(camera_id, analysis)
                    coordination_result = self.process_detection_message(message)
                    
                    coordination_results['messages_sent'] += 1
                    coordination_results['actions_taken'].extend(coordination_result['actions'])
                    
                    if coordination_result['actions']:
                        print(f"  {camera_id}: {coordination_result['actions']}")
        
        # Update incident counts
        coordination_results['incidents_created'] = len([i for i in self.active_incidents.values() 
                                                       if len(i.cameras_involved) == 1])
        coordination_results['incidents_updated'] = len([i for i in self.active_incidents.values() 
                                                        if len(i.cameras_involved) > 1])
        
        return coordination_results
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            'active_incidents': len(self.active_incidents),
            'total_messages': len(self.message_queue),
            'cameras_online': len(self.camera_states),
            'incidents': {
                incident_id: {
                    'threat_type': incident.threat_type,
                    'threat_level': incident.threat_level,
                    'cameras_involved': incident.cameras_involved,
                    'status': incident.status,
                    'duration': time.time() - incident.start_time
                }
                for incident_id, incident in self.active_incidents.items()
            }
        }
    
    def get_active_incidents_summary(self) -> List[Dict[str, Any]]:
        """Get summary of active incidents"""
        incidents = []
        
        for incident_id, incident in self.active_incidents.items():
            incidents.append({
                'incident_id': incident_id,
                'threat_type': incident.threat_type,
                'threat_level': incident.threat_level,
                'cameras_involved': incident.cameras_involved,
                'status': incident.status,
                'duration_seconds': time.time() - incident.start_time,
                'evidence_count': len(incident.evidence)
            })
        
        return incidents

def test_mcp_coordination():
    """Test the MCP coordination system"""
    print("Testing MCP Coordination System...")
    
    hub = MCPCoordinationHub()
    
    # Test single coordination cycle
    print("\nRunning Coordination Cycle:")
    results = hub.coordinate_camera_feeds()
    
    print(f"  Cameras processed: {results['cameras_processed']}")
    print(f"  Messages sent: {results['messages_sent']}")
    print(f"  Incidents created: {results['incidents_created']}")
    print(f"  Incidents updated: {results['incidents_updated']}")
    print(f"  Actions taken: {len(results['actions_taken'])}")
    
    # Test coordination status
    print("\nCoordination Status:")
    status = hub.get_coordination_status()
    print(f"  Active incidents: {status['active_incidents']}")
    print(f"  Total messages: {status['total_messages']}")
    print(f"  Cameras online: {status['cameras_online']}")
    
    # Test incident summary
    print("\nActive Incidents Summary:")
    incidents = hub.get_active_incidents_summary()
    for incident in incidents:
        print(f"  {incident['incident_id']}: {incident['threat_type']} ({incident['threat_level']}) - {len(incident['cameras_involved'])} cameras")
    
    print("\nMCP coordination test completed!")

if __name__ == "__main__":
    test_mcp_coordination()