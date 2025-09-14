"""
Evidence Collection System
Automatically saves video clips when threats are detected
"""

import os
import json
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

class EvidenceCollector:
    """Evidence collection system for threat detection"""
    
    def __init__(self, evidence_dir: str = "evidence"):
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.video_clips_dir = self.evidence_dir / "video_clips"
        self.metadata_dir = self.evidence_dir / "metadata"
        self.screenshots_dir = self.evidence_dir / "screenshots"
        
        for dir_path in [self.video_clips_dir, self.metadata_dir, self.screenshots_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Evidence database
        self.evidence_db_file = self.evidence_dir / "evidence_database.json"
        self.evidence_db = self.load_evidence_database()
        
        # Video processing settings
        self.clip_duration_before = 10  # seconds before detection
        self.clip_duration_after = 10   # seconds after detection
        self.max_clip_duration = 60      # maximum clip length
        
        print(f"✅ Evidence collection system initialized")
        print(f"📁 Evidence directory: {self.evidence_dir.absolute()}")
    
    def load_evidence_database(self) -> Dict:
        """Load evidence database from JSON file"""
        if self.evidence_db_file.exists():
            try:
                with open(self.evidence_db_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading evidence database: {e}")
                return {"incidents": [], "metadata": {}}
        return {"incidents": [], "metadata": {}}
    
    def save_evidence_database(self):
        """Save evidence database to JSON file"""
        try:
            with open(self.evidence_db_file, 'w') as f:
                json.dump(self.evidence_db, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving evidence database: {e}")
    
    def generate_evidence_id(self, detection: Dict) -> str:
        """Generate unique evidence ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        camera_id = detection.get('camera_id', 'unknown')
        threat_type = detection.get('threat_type', 'unknown').replace(' ', '_')
        return f"{threat_type}_{camera_id}_{timestamp}"
    
    def extract_video_clip(self, source_video_path: str, detection: Dict) -> Optional[str]:
        """Extract video clip around detection timestamp"""
        try:
            # Get video info
            cap = cv2.VideoCapture(source_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            if duration == 0:
                print(f"❌ Invalid video duration: {source_video_path}")
                return None
            
            # Calculate clip timing
            detection_time = self.parse_detection_timestamp(detection)
            start_time = max(0, detection_time - self.clip_duration_before)
            end_time = min(duration, detection_time + self.clip_duration_after)
            
            # Ensure minimum clip duration
            if end_time - start_time < 5:
                center = (start_time + end_time) / 2
                start_time = max(0, center - 5)
                end_time = min(duration, center + 5)
            
            # Generate output filename
            evidence_id = self.generate_evidence_id(detection)
            output_path = self.video_clips_dir / f"{evidence_id}.mp4"
            
            # Extract clip using FFmpeg
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', source_video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',  # Copy without re-encoding for speed
                '-avoid_negative_ts', 'make_zero',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_path.exists():
                print(f"✅ Video clip extracted: {output_path.name}")
                return str(output_path)
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting video clip: {e}")
            return None
    
    def parse_detection_timestamp(self, detection: Dict) -> float:
        """Parse detection timestamp to seconds"""
        timestamp_str = detection.get('timestamp', '')
        
        # Try different timestamp formats
        try:
            if ':' in timestamp_str:
                # Format: "16:45:30" or "16:45:30.123"
                time_parts = timestamp_str.split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds = float(time_parts[2])
                return hours * 3600 + minutes * 60 + seconds
            else:
                # Assume it's already in seconds
                return float(timestamp_str)
        except:
            # Default to middle of video if parsing fails
            return 30.0
    
    def create_evidence_metadata(self, detection: Dict, video_clip_path: Optional[str] = None) -> Dict:
        """Create metadata for evidence"""
        evidence_id = self.generate_evidence_id(detection)
        
        metadata = {
            "evidence_id": evidence_id,
            "incident_id": f"inc_{evidence_id}",
            "timestamp": datetime.now().isoformat(),
            "detection": {
                "camera_id": detection.get('camera_id', 'unknown'),
                "threat_type": detection.get('threat_type', 'unknown'),
                "confidence": detection.get('confidence', 0.0),
                "severity": detection.get('severity', 'unknown'),
                "indicators": detection.get('indicators', []),
                "location": detection.get('location', {}),
                "detection_timestamp": detection.get('timestamp', '')
            },
            "evidence": {
                "video_clip_path": video_clip_path,
                "video_clip_size": os.path.getsize(video_clip_path) if video_clip_path and os.path.exists(video_clip_path) else 0,
                "clip_duration_before": self.clip_duration_before,
                "clip_duration_after": self.clip_duration_after,
                "extraction_method": "ffmpeg"
            },
            "system": {
                "collection_timestamp": datetime.now().isoformat(),
                "evidence_collector_version": "1.0",
                "processing_status": "completed" if video_clip_path else "failed"
            }
        }
        
        return metadata
    
    def save_evidence_metadata(self, metadata: Dict):
        """Save evidence metadata to file"""
        try:
            evidence_id = metadata["evidence_id"]
            metadata_file = self.metadata_dir / f"{evidence_id}.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Evidence metadata saved: {metadata_file.name}")
            
        except Exception as e:
            print(f"❌ Error saving evidence metadata: {e}")
    
    def collect_evidence(self, detection: Dict, source_video_path: Optional[str] = None) -> Dict:
        """Main evidence collection function"""
        print(f"🔍 Collecting evidence for {detection.get('threat_type')} detection...")
        
        evidence_id = self.generate_evidence_id(detection)
        
        # Extract video clip if source video is provided
        video_clip_path = None
        if source_video_path and os.path.exists(source_video_path):
            video_clip_path = self.extract_video_clip(source_video_path, detection)
        else:
            print(f"⚠️ No source video provided or file doesn't exist: {source_video_path}")
        
        # Create metadata
        metadata = self.create_evidence_metadata(detection, video_clip_path)
        
        # Save metadata
        self.save_evidence_metadata(metadata)
        
        # Add to evidence database
        self.evidence_db["incidents"].append(metadata)
        self.evidence_db["metadata"]["total_incidents"] = len(self.evidence_db["incidents"])
        self.evidence_db["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save database
        self.save_evidence_database()
        
        print(f"✅ Evidence collected: {evidence_id}")
        
        return metadata
    
    def get_evidence_summary(self) -> Dict:
        """Get summary of all collected evidence"""
        incidents = self.evidence_db.get("incidents", [])
        
        summary = {
            "total_incidents": len(incidents),
            "evidence_directory": str(self.evidence_dir.absolute()),
            "video_clips_count": len([inc for inc in incidents if inc.get("evidence", {}).get("video_clip_path")]),
            "total_storage_used": self.calculate_storage_usage(),
            "recent_incidents": incidents[-5:] if incidents else [],
            "threat_types": self.get_threat_type_summary(incidents),
            "cameras_involved": self.get_camera_summary(incidents)
        }
        
        return summary
    
    def calculate_storage_usage(self) -> Dict:
        """Calculate storage usage of evidence files"""
        total_size = 0
        file_counts = {"video_clips": 0, "metadata": 0, "screenshots": 0}
        
        for dir_name, dir_path in [
            ("video_clips", self.video_clips_dir),
            ("metadata", self.metadata_dir),
            ("screenshots", self.screenshots_dir)
        ]:
            if dir_path.exists():
                for file_path in dir_path.iterdir():
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_counts[dir_name] += 1
        
        return {
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2),
            "file_counts": file_counts
        }
    
    def get_threat_type_summary(self, incidents: List[Dict]) -> Dict:
        """Get summary of threat types"""
        threat_counts = {}
        for incident in incidents:
            threat_type = incident.get("detection", {}).get("threat_type", "unknown")
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        return threat_counts
    
    def get_camera_summary(self, incidents: List[Dict]) -> Dict:
        """Get summary of cameras involved"""
        camera_counts = {}
        for incident in incidents:
            camera_id = incident.get("detection", {}).get("camera_id", "unknown")
            camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
        return camera_counts
    
    def cleanup_old_evidence(self, days_to_keep: int = 30):
        """Clean up evidence older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        removed_count = 0
        
        for incident in self.evidence_db.get("incidents", []):
            try:
                incident_date = datetime.fromisoformat(incident.get("timestamp", ""))
                if incident_date < cutoff_date:
                    # Remove video clip
                    video_path = incident.get("evidence", {}).get("video_clip_path")
                    if video_path and os.path.exists(video_path):
                        os.remove(video_path)
                        removed_count += 1
                    
                    # Remove metadata file
                    evidence_id = incident.get("evidence_id")
                    if evidence_id:
                        metadata_file = self.metadata_dir / f"{evidence_id}.json"
                        if metadata_file.exists():
                            metadata_file.unlink()
            
            except Exception as e:
                print(f"❌ Error cleaning up evidence: {e}")
        
        print(f"🧹 Cleaned up {removed_count} old evidence files")
    
    def get_evidence_by_id(self, evidence_id: str) -> Optional[Dict]:
        """Get specific evidence by ID"""
        for incident in self.evidence_db.get("incidents", []):
            if incident.get("evidence_id") == evidence_id:
                return incident
        return None
    
    def list_evidence_files(self) -> List[Dict]:
        """List all evidence files with metadata"""
        evidence_files = []
        
        for incident in self.evidence_db.get("incidents", []):
            evidence_id = incident.get("evidence_id")
            video_path = incident.get("evidence", {}).get("video_clip_path")
            
            if evidence_id and video_path and os.path.exists(video_path):
                file_info = {
                    "evidence_id": evidence_id,
                    "filename": os.path.basename(video_path),
                    "file_path": video_path,
                    "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
                    "threat_type": incident.get("detection", {}).get("threat_type"),
                    "camera_id": incident.get("detection", {}).get("camera_id"),
                    "timestamp": incident.get("timestamp"),
                    "confidence": incident.get("detection", {}).get("confidence")
                }
                evidence_files.append(file_info)
        
        return evidence_files

# Global evidence collector instance
evidence_collector = EvidenceCollector()

def collect_evidence_for_detection(detection: Dict, source_video_path: Optional[str] = None) -> Dict:
    """Collect evidence for a detection"""
    return evidence_collector.collect_evidence(detection, source_video_path)

def get_evidence_summary() -> Dict:
    """Get evidence collection summary"""
    return evidence_collector.get_evidence_summary()

def list_evidence_files() -> List[Dict]:
    """List all evidence files"""
    return evidence_collector.list_evidence_files()

def get_evidence_by_id(evidence_id: str) -> Optional[Dict]:
    """Get evidence by ID"""
    return evidence_collector.get_evidence_by_id(evidence_id)

# Test function
def test_evidence_collection():
    """Test evidence collection system"""
    print("🧪 Testing Evidence Collection System...")
    
    # Sample detection
    sample_detection = {
        "camera_id": "cam_1",
        "threat_type": "fighting",
        "confidence": 0.95,
        "timestamp": "16:45:30",
        "severity": "high",
        "indicators": ["High motion detected", "Rapid color changes"],
        "location": {"x": 150, "y": 200}
    }
    
    # Test evidence collection
    print("\n📹 Testing evidence collection...")
    metadata = collect_evidence_for_detection(sample_detection)
    
    # Test summary
    print("\n📊 Evidence summary:")
    summary = get_evidence_summary()
    print(f"  Total incidents: {summary['total_incidents']}")
    print(f"  Storage used: {summary['total_storage_used']['total_mb']} MB")
    print(f"  Threat types: {summary['threat_types']}")
    
    # Test file listing
    print("\n📁 Evidence files:")
    files = list_evidence_files()
    for file_info in files:
        print(f"  {file_info['filename']} - {file_info['threat_type']} ({file_info['file_size_mb']} MB)")
    
    print("\n✅ Evidence collection test completed!")

if __name__ == "__main__":
    test_evidence_collection()
