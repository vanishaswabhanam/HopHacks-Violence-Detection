"""
Redis-Based MCP Coordination System
High-performance distributed camera coordination using Redis Pub/Sub
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import aioredis
from multi_camera_system import MultiCameraSystem

@dataclass
class DetectionMessage:
    """Redis message format for camera detections"""
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
    message_type: str = "detection"

@dataclass
class CoordinationMessage:
    """Redis message format for coordination actions"""
    message_id: str
    action_type: str  # "merge_incident", "escalate", "resolve"
    target_incident: str
    source_camera: str
    timestamp: float
    metadata: Dict[str, Any]

class RedisMCPCoordinationHub:
    """Redis-based Multi-Camera Protocol Coordination Hub"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.camera_system = MultiCameraSystem()
        
        # Redis channels
        self.detection_channel = "mcp:detections"
        self.coordination_channel = "mcp:coordination"
        self.status_channel = "mcp:status"
        
        # State management
        self.active_incidents = {}
        self.camera_states = {}
        self.message_history = []
        
        # Coordination rules
        self.coordination_rules = {
            'duplicate_threshold': 0.8,
            'escalation_threshold': 0.9,
            'time_window': 30,
            'max_cameras_per_incident': 4,
            'message_ttl': 3600  # 1 hour
        }
    
    async def connect_redis(self):
        """Connect to Redis server"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to coordination channels
            await self.pubsub.subscribe(
                self.detection_channel,
                self.coordination_channel,
                self.status_channel
            )
            
            print("✅ Connected to Redis MCP Coordination Hub")
            return True
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False
    
    async def disconnect_redis(self):
        """Disconnect from Redis"""
        if self.pubsub:
            await self.pubsub.unsubscribe()
        if self.redis_client:
            await self.redis_client.close()
        print("🔌 Disconnected from Redis")
    
    def generate_incident_id(self) -> str:
        """Generate unique incident ID with timestamp"""
        return f"inc_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    async def publish_detection(self, camera_id: str, analysis_result: Dict[str, Any]) -> str:
        """Publish detection message to Redis"""
        incident_id = self.generate_incident_id()
        
        message = DetectionMessage(
            message_id=self.generate_message_id(),
            camera_id=camera_id,
            incident_id=incident_id,
            threat_type=analysis_result.get('threat_type', 'unknown'),
            threat_level=analysis_result.get('threat_level', 'low'),
            confidence=analysis_result.get('confidence', 0.0),
            location={'x': 150, 'y': 200},
            timestamp=time.time(),
            indicators=analysis_result.get('indicators', []),
            multi_class_analysis=analysis_result.get('multi_class_analysis')
        )
        
        # Serialize and publish
        message_data = json.dumps(asdict(message))
        await self.redis_client.publish(self.detection_channel, message_data)
        
        # Store in message history with TTL
        await self.redis_client.setex(
            f"mcp:message:{message.message_id}",
            self.coordination_rules['message_ttl'],
            message_data
        )
        
        print(f"📡 Published detection from {camera_id}: {message.threat_type} ({message.confidence:.2f})")
        return message.message_id
    
    async def publish_coordination_action(self, action_type: str, target_incident: str, 
                                       source_camera: str, metadata: Dict[str, Any] = None):
        """Publish coordination action to Redis"""
        message = CoordinationMessage(
            message_id=self.generate_message_id(),
            action_type=action_type,
            target_incident=target_incident,
            source_camera=source_camera,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        message_data = json.dumps(asdict(message))
        await self.redis_client.publish(self.coordination_channel, message_data)
        
        print(f"🎯 Published coordination action: {action_type} for {target_incident}")
    
    async def process_detection_message(self, message_data: str) -> Dict[str, Any]:
        """Process incoming detection message"""
        try:
            message_dict = json.loads(message_data)
            message = DetectionMessage(**message_dict)
            
            actions = []
            
            # Check for duplicate incidents
            duplicate_incident = await self.check_for_duplicate_incident(message)
            if duplicate_incident:
                await self.publish_coordination_action(
                    "merge_incident",
                    duplicate_incident,
                    message.camera_id,
                    {"original_incident": message.incident_id}
                )
                message.incident_id = duplicate_incident
                actions.append("merged_with_existing")
            
            # Update incident state
            await self.update_incident_state(message)
            
            # Check for escalation
            if message.confidence > self.coordination_rules['escalation_threshold']:
                await self.publish_coordination_action(
                    "escalate_incident",
                    message.incident_id,
                    message.camera_id,
                    {"confidence": message.confidence}
                )
                actions.append("escalated")
            
            # Update camera state
            await self.update_camera_state(message)
            
            return {
                'message_id': message.message_id,
                'incident_id': message.incident_id,
                'actions': actions,
                'status': 'processed'
            }
            
        except Exception as e:
            print(f"❌ Error processing detection message: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def check_for_duplicate_incident(self, message: DetectionMessage) -> Optional[str]:
        """Check Redis for duplicate incidents using pattern matching"""
        current_time = message.timestamp
        
        # Get all incident keys
        incident_keys = await self.redis_client.keys("mcp:incident:*")
        
        for key in incident_keys:
            incident_data = await self.redis_client.get(key)
            if incident_data:
                incident = json.loads(incident_data)
                
                # Check time window
                if current_time - incident['start_time'] > self.coordination_rules['time_window']:
                    continue
                
                # Check threat type and camera adjacency
                if (incident['threat_type'] == message.threat_type and 
                    self.are_cameras_adjacent(incident['cameras_involved'][-1], message.camera_id)):
                    return incident['incident_id']
        
        return None
    
    async def update_incident_state(self, message: DetectionMessage):
        """Update incident state in Redis"""
        incident_key = f"mcp:incident:{message.incident_id}"
        
        # Check if incident exists
        existing_data = await self.redis_client.get(incident_key)
        
        if existing_data:
            # Update existing incident
            incident = json.loads(existing_data)
            if message.camera_id not in incident['cameras_involved']:
                incident['cameras_involved'].append(message.camera_id)
            incident['evidence'].append(asdict(message))
            incident['last_update'] = message.timestamp
        else:
            # Create new incident
            incident = {
                'incident_id': message.incident_id,
                'primary_camera': message.camera_id,
                'threat_type': message.threat_type,
                'threat_level': message.threat_level,
                'start_time': message.timestamp,
                'last_update': message.timestamp,
                'cameras_involved': [message.camera_id],
                'status': 'active',
                'evidence': [asdict(message)]
            }
        
        # Store in Redis with TTL
        await self.redis_client.setex(
            incident_key,
            self.coordination_rules['message_ttl'],
            json.dumps(incident)
        )
    
    async def update_camera_state(self, message: DetectionMessage):
        """Update camera state in Redis"""
        camera_key = f"mcp:camera:{message.camera_id}"
        
        camera_state = {
            'camera_id': message.camera_id,
            'last_detection': message.timestamp,
            'current_incident': message.incident_id,
            'threat_level': message.threat_level,
            'status': 'active'
        }
        
        await self.redis_client.setex(
            camera_key,
            self.coordination_rules['message_ttl'],
            json.dumps(camera_state)
        )
    
    def are_cameras_adjacent(self, camera1: str, camera2: str) -> bool:
        """Check camera adjacency using building topology"""
        adjacency_map = {
            'cam_1': ['cam_2', 'cam_3'],  # Hallway A -> Cafeteria, Main Entrance
            'cam_2': ['cam_1', 'cam_4'],  # Cafeteria -> Hallway A, Gymnasium
            'cam_3': ['cam_1', 'cam_4'],  # Main Entrance -> Hallway A, Gymnasium
            'cam_4': ['cam_2', 'cam_3']   # Gymnasium -> Cafeteria, Main Entrance
        }
        return camera2 in adjacency_map.get(camera1, [])
    
    async def coordinate_camera_feeds(self) -> Dict[str, Any]:
        """Coordinate all camera feeds using Redis pub/sub"""
        print("🚀 Starting Redis MCP Coordination...")
        
        # Process all camera feeds
        camera_results = self.camera_system.process_all_cameras()
        
        coordination_results = {
            'timestamp': time.time(),
            'cameras_processed': len(camera_results),
            'messages_published': 0,
            'incidents_created': 0,
            'coordination_actions': []
        }
        
        # Publish detections to Redis
        for camera_id, result in camera_results.items():
            if result.get('analysis', {}).get('status') == 'success':
                analysis = result['analysis']
                
                if analysis.get('confidence', 0) > 0.5:
                    message_id = await self.publish_detection(camera_id, analysis)
                    coordination_results['messages_published'] += 1
        
        # Process coordination messages
        await self.process_coordination_messages()
        
        # Get incident statistics
        incident_keys = await self.redis_client.keys("mcp:incident:*")
        coordination_results['incidents_created'] = len(incident_keys)
        
        return coordination_results
    
    async def process_coordination_messages(self):
        """Process incoming coordination messages"""
        try:
            # Get messages from coordination channel
            messages = await self.redis_client.lrange("mcp:coordination:queue", 0, -1)
            
            for message_data in messages:
                message_dict = json.loads(message_data)
                message = CoordinationMessage(**message_dict)
                
                print(f"🎯 Processing coordination: {message.action_type} for {message.target_incident}")
                
                # Process different action types
                if message.action_type == "merge_incident":
                    await self.merge_incidents(message)
                elif message.action_type == "escalate":
                    await self.escalate_incident(message)
                elif message.action_type == "resolve":
                    await self.resolve_incident(message)
        
        except Exception as e:
            print(f"❌ Error processing coordination messages: {e}")
    
    async def merge_incidents(self, message: CoordinationMessage):
        """Merge two incidents"""
        # Implementation for merging incidents
        print(f"🔄 Merging incidents: {message.target_incident}")
    
    async def escalate_incident(self, message: CoordinationMessage):
        """Escalate incident severity"""
        incident_key = f"mcp:incident:{message.target_incident}"
        incident_data = await self.redis_client.get(incident_key)
        
        if incident_data:
            incident = json.loads(incident_data)
            incident['status'] = 'escalated'
            incident['escalation_time'] = message.timestamp
            
            await self.redis_client.setex(
                incident_key,
                self.coordination_rules['message_ttl'],
                json.dumps(incident)
            )
            print(f"🚨 Escalated incident: {message.target_incident}")
    
    async def resolve_incident(self, message: CoordinationMessage):
        """Resolve incident"""
        incident_key = f"mcp:incident:{message.target_incident}"
        incident_data = await self.redis_client.get(incident_key)
        
        if incident_data:
            incident = json.loads(incident_data)
            incident['status'] = 'resolved'
            incident['resolution_time'] = message.timestamp
            
            await self.redis_client.setex(
                incident_key,
                self.coordination_rules['message_ttl'],
                json.dumps(incident)
            )
            print(f"✅ Resolved incident: {message.target_incident}")
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status from Redis"""
        # Get all incident keys
        incident_keys = await self.redis_client.keys("mcp:incident:*")
        camera_keys = await self.redis_client.keys("mcp:camera:*")
        
        incidents = []
        for key in incident_keys:
            incident_data = await self.redis_client.get(key)
            if incident_data:
                incidents.append(json.loads(incident_data))
        
        cameras = []
        for key in camera_keys:
            camera_data = await self.redis_client.get(key)
            if camera_data:
                cameras.append(json.loads(camera_data))
        
        return {
            'active_incidents': len(incidents),
            'cameras_online': len(cameras),
            'redis_connected': True,
            'incidents': incidents,
            'cameras': cameras,
            'coordination_rules': self.coordination_rules
        }
    
    async def get_active_incidents_summary(self) -> List[Dict[str, Any]]:
        """Get summary of active incidents from Redis"""
        incident_keys = await self.redis_client.keys("mcp:incident:*")
        incidents = []
        
        for key in incident_keys:
            incident_data = await self.redis_client.get(key)
            if incident_data:
                incident = json.loads(incident_data)
                incidents.append({
                    'incident_id': incident['incident_id'],
                    'threat_type': incident['threat_type'],
                    'threat_level': incident['threat_level'],
                    'cameras_involved': incident['cameras_involved'],
                    'status': incident['status'],
                    'duration_seconds': time.time() - incident['start_time'],
                    'evidence_count': len(incident['evidence'])
                })
        
        return incidents

async def test_redis_mcp_coordination():
    """Test the Redis-based MCP coordination system"""
    print("🧪 Testing Redis MCP Coordination System...")
    
    hub = RedisMCPCoordinationHub()
    
    # Connect to Redis
    if not await hub.connect_redis():
        print("❌ Failed to connect to Redis")
        return
    
    try:
        # Test coordination cycle
        print("\n🚀 Running Redis Coordination Cycle:")
        results = await hub.coordinate_camera_feeds()
        
        print(f"  Cameras processed: {results['cameras_processed']}")
        print(f"  Messages published: {results['messages_published']}")
        print(f"  Incidents created: {results['incidents_created']}")
        
        # Test status
        print("\n📊 Coordination Status:")
        status = await hub.get_coordination_status()
        print(f"  Active incidents: {status['active_incidents']}")
        print(f"  Cameras online: {status['cameras_online']}")
        print(f"  Redis connected: {status['redis_connected']}")
        
        # Test incident summary
        print("\n📋 Active Incidents Summary:")
        incidents = await hub.get_active_incidents_summary()
        for incident in incidents:
            print(f"  {incident['incident_id']}: {incident['threat_type']} ({incident['threat_level']}) - {len(incident['cameras_involved'])} cameras")
    
    finally:
        await hub.disconnect_redis()
    
    print("\n✅ Redis MCP coordination test completed!")

if __name__ == "__main__":
    asyncio.run(test_redis_mcp_coordination())