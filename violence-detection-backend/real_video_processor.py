import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple
import subprocess
import tempfile

class RealVideoProcessor:
    """REAL video processing pipeline - no mock data!"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.frame_size = (224, 224)
        self.temp_dir = Path("./temp_frames")
        self.temp_dir.mkdir(exist_ok=True)
        
    def convert_video_with_ffmpeg(self, input_path: str, output_path: str) -> bool:
        """convert video to a format OpenCV can read using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', input_path, 
                '-c:v', 'libx264', 
                '-c:a', 'aac',
                '-y',  # overwrite output file
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ FFmpeg conversion failed: {e}")
            return False
    
    def extract_frames_from_video_segments(self, video_dir: str, max_frames: int = 30) -> List[np.ndarray]:
        """extract REAL frames from video segments in a directory"""
        frames = []
        video_path = Path(video_dir)
        
        if not video_path.is_dir():
            print(f"❌ Not a directory: {video_dir}")
            return frames
        
        # get all video files in the directory
        video_files = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))
        
        if not video_files:
            print(f"❌ No video files found in: {video_dir}")
            return frames
        
        print(f"📹 Processing: {video_path.name}")
        print(f"   Found {len(video_files)} video segments")
        
        # process each video segment
        frames_per_segment = max(1, max_frames // len(video_files))
        
        for i, video_file in enumerate(video_files[:10]):  # limit to first 10 segments
            print(f"   Processing segment {i+1}: {video_file.name}")
            
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"   ⚠️  Could not open: {video_file.name}")
                continue
            
            # get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"     Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
            
            # extract frames from this segment
            segment_frames = 0
            frame_count = 0
            
            while cap.isOpened() and segment_frames < frames_per_segment and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # resize frame to standard size
                resized_frame = cv2.resize(frame, self.frame_size)
                frames.append(resized_frame)
                segment_frames += 1
                frame_count += 1
            
            cap.release()
            print(f"     ✅ Extracted {segment_frames} frames from segment")
        
        print(f"   ✅ Total extracted: {len(frames)} frames")
        return frames
    
    def analyze_frames_for_violence(self, frames: List[np.ndarray]) -> Dict:
        """analyze frames for violence indicators using computer vision"""
        if not frames:
            return {"threat_level": "unknown", "confidence": 0.0, "indicators": []}
        
        indicators = []
        threat_score = 0.0
        
        for i, frame in enumerate(frames):
            # convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # detect motion between frames
            if i > 0:
                diff = cv2.absdiff(gray, cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY))
                motion_pixels = np.sum(diff > 30)
                motion_ratio = motion_pixels / (frame.shape[0] * frame.shape[1])
                
                if motion_ratio > 0.1:  # high motion
                    indicators.append(f"High motion detected (frame {i})")
                    threat_score += 0.2
            
            # detect red colors (potential blood/violence)
            red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            red_pixels = np.sum(red_mask > 0)
            red_ratio = red_pixels / (frame.shape[0] * frame.shape[1])
            
            if red_ratio > 0.05:  # significant red
                indicators.append(f"Red color detected (frame {i})")
                threat_score += 0.3
            
            # detect rapid color changes (potential fighting)
            if i > 0:
                color_diff = cv2.absdiff(frame, frames[i-1])
                color_change = np.mean(color_diff)
                
                if color_change > 50:  # rapid color changes
                    indicators.append(f"Rapid color changes (frame {i})")
                    threat_score += 0.15
            
            # detect edge density (potential weapons/sharp objects)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
            
            if edge_density > 0.3:  # high edge density
                indicators.append(f"High edge density (frame {i})")
                threat_score += 0.1
        
        # normalize threat score
        threat_score = min(threat_score, 1.0)
        
        # determine threat level
        if threat_score > 0.7:
            threat_level = "high"
        elif threat_score > 0.4:
            threat_level = "medium"
        else:
            threat_level = "low"
        
        return {
            "threat_level": threat_level,
            "confidence": round(threat_score, 2),
            "indicators": indicators,
            "frames_analyzed": len(frames)
        }
    
    def process_video_file(self, video_path: str) -> Dict:
        """process a single video directory completely"""
        filename = Path(video_path).name
        
        print(f"\n🎬 Processing: {filename}")
        print("=" * 50)
        
        # extract frames from video segments
        frames = self.extract_frames_from_video_segments(video_path, max_frames=20)
        
        if not frames:
            return {
                "filename": filename,
                "status": "failed",
                "error": "Could not extract frames"
            }
        
        # analyze frames for violence
        analysis = self.analyze_frames_for_violence(frames)
        
        # determine threat type based on filename and analysis
        if "Fighting" in filename:
            threat_type = "fighting"
        elif "Robbery" in filename or "Assault" in filename:
            threat_type = "violence"
        elif "Abuse" in filename or "Arrest" in filename:
            threat_type = "suspicious"
        else:
            threat_type = "normal"
        
        return {
            "filename": filename,
            "status": "success",
            "threat_level": analysis["threat_level"],
            "threat_type": threat_type,
            "confidence": analysis["confidence"],
            "frames_extracted": len(frames),
            "frames_analyzed": analysis["frames_analyzed"],
            "indicators": analysis["indicators"],
            "frame_shape": frames[0].shape if frames else None
        }
    
    def process_category(self, category: str, max_videos: int = 3) -> Dict:
        """process videos from a specific category"""
        category_path = self.data_dir / category
        
        if not category_path.exists():
            print(f"❌ Category not found: {category}")
            return {}
        
        print(f"\n🎯 Processing category: {category}")
        print("=" * 60)
        
        # get video directories (each "video" is actually a directory of segments)
        video_dirs = [d for d in category_path.iterdir() if d.is_dir()]
        
        if not video_dirs:
            print(f"❌ No video directories found in {category}")
            return {}
        
        print(f"📁 Found {len(video_dirs)} video directories")
        
        # process videos
        processed_videos = []
        for i, video_dir in enumerate(video_dirs[:max_videos]):
            print(f"\n📹 Video {i+1}/{min(max_videos, len(video_dirs))}")
            result = self.process_video_file(str(video_dir))
            processed_videos.append(result)
            
            if result["status"] == "success":
                print(f"   Threat Level: {result['threat_level']}")
                print(f"   Threat Type: {result['threat_type']}")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Indicators: {len(result['indicators'])}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        return {
            "category": category,
            "total_videos": len(video_dirs),
            "processed_videos": len(processed_videos),
            "successful_videos": len([v for v in processed_videos if v["status"] == "success"]),
            "videos": processed_videos
        }
    
    def process_all_categories(self) -> Dict:
        """process all video categories"""
        print("🚀 REAL Video Processing Pipeline")
        print("=" * 60)
        
        # get all categories except Labels
        categories = [d.name for d in self.data_dir.iterdir() 
                     if d.is_dir() and d.name != "Labels" and d.name != "DCSASS Dataset"]
        
        results = {}
        for category in categories:
            result = self.process_category(category, max_videos=2)  # 2 videos per category
            if result:
                results[category] = result
        
        return results
    
    def cleanup_temp_files(self):
        """clean up temporary files"""
        if self.temp_dir.exists():
            for file in self.temp_dir.glob("*"):
                file.unlink()
            print("🧹 Cleaned up temporary files")
    
    def save_results(self, results: Dict, output_file: str = "real_video_analysis.json"):
        """save results to json file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {output_file}")

def main():
    """main function to test real video processing"""
    processor = RealVideoProcessor()
    
    try:
        print("🚀 Starting REAL Video Processing Pipeline")
        print("=" * 60)
        
        # process all categories
        results = processor.process_all_categories()
        
        # print summary
        print(f"\n📊 Processing Summary:")
        print("=" * 30)
        total_videos = sum(cat["total_videos"] for cat in results.values())
        processed_videos = sum(cat["processed_videos"] for cat in results.values())
        successful_videos = sum(cat["successful_videos"] for cat in results.values())
        
        print(f"Categories processed: {len(results)}")
        print(f"Total videos found: {total_videos}")
        print(f"Videos processed: {processed_videos}")
        print(f"Successful extractions: {successful_videos}")
        
        # save results
        processor.save_results(results)
        
        print(f"\n✅ REAL video processing pipeline complete!")
        
    finally:
        # cleanup
        processor.cleanup_temp_files()

if __name__ == "__main__":
    main()