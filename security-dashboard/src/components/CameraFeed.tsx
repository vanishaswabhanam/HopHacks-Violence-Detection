import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle, Play, Square } from 'lucide-react';
import './CameraFeed.css';

interface Detection {
  camera_id: string;
  threat_type: string;
  confidence: number;
  timestamp: string;
  severity: string;
  indicators: string[];
}

interface StreamInfo {
  video_path: string;
  frame_count: number;
  fps: number;
  width: number;
  height: number;
  violence_detected: boolean;
  last_detection: number | null;
  running: boolean;
}

interface CameraFeedProps {
  name: string;
  cameraId: string;
  category?: string;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ name, cameraId, category = 'Fighting' }) => {
  const [detection, setDetection] = useState<Detection | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [streamInfo, setStreamInfo] = useState<StreamInfo | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [videoFrame, setVideoFrame] = useState<string | null>(null);

  useEffect(() => {
    // update time every second
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);

    // fetch detections for this camera
    const fetchDetection = async () => {
      try {
        const response = await fetch('http://localhost:8000/detections');
        const data = await response.json();
        const cameraDetection = data.detections?.find((d: Detection) => d.camera_id === cameraId);
        setDetection(cameraDetection || null);
      } catch (error) {
        console.error('Failed to fetch detection:', error);
      }
    };

    // fetch stream info
    const fetchStreamInfo = async () => {
      try {
        const response = await fetch(`http://localhost:8000/video/stream/${cameraId}`);
        const data = await response.json();
        if (data.status === 'success') {
          setStreamInfo(data.stream_info);
          setIsStreaming(data.stream_info.running);
        } else {
          setIsStreaming(false);
        }
      } catch (error) {
        console.error('Failed to fetch stream info:', error);
        setIsStreaming(false);
      }
    };

    // fetch video frame
    const fetchVideoFrame = async () => {
      if (!isStreaming) return;
      
      try {
        const response = await fetch(`http://localhost:8000/video/frame/${cameraId}`);
        if (response.ok) {
          const blob = await response.blob();
          const imageUrl = URL.createObjectURL(blob);
          setVideoFrame(imageUrl);
        }
      } catch (error) {
        console.error('Failed to fetch video frame:', error);
      }
    };

    fetchDetection();
    fetchStreamInfo();
    fetchVideoFrame();
    
    const detectionInterval = setInterval(fetchDetection, 3000); // refresh every 3 seconds
    const streamInterval = setInterval(fetchStreamInfo, 2000); // refresh every 2 seconds
    const frameInterval = setInterval(fetchVideoFrame, 100); // refresh every 100ms for smooth video

    return () => {
      clearInterval(timeInterval);
      clearInterval(detectionInterval);
      clearInterval(streamInterval);
      clearInterval(frameInterval);
    };
  }, [cameraId, isStreaming]);

  const hasAlert = detection && detection.confidence > 0.5;
  const hasViolence = streamInfo?.violence_detected || false;

  const startStream = async () => {
    try {
      setStreamError(null);
      const response = await fetch(`http://localhost:8000/video/stream/start/${cameraId}?category=${category}`, {
        method: 'POST'
      });
      const data = await response.json();
      if (data.status === 'success') {
        setIsStreaming(true);
      } else {
        setStreamError(data.message);
      }
    } catch (error) {
      setStreamError('Failed to start stream');
      console.error('Failed to start stream:', error);
    }
  };

  const stopStream = async () => {
    try {
      const response = await fetch(`http://localhost:8000/video/stream/stop/${cameraId}`, {
        method: 'POST'
      });
      const data = await response.json();
      if (data.status === 'success') {
        setIsStreaming(false);
        setStreamInfo(null);
      }
    } catch (error) {
      console.error('Failed to stop stream:', error);
    }
  };

  return (
    <div className={`camera-feed ${hasAlert || hasViolence ? 'alert' : ''}`}>
      <div className="camera-header">
        <span className="camera-name">{name}</span>
        <div className="stream-controls">
          {!isStreaming ? (
            <button onClick={startStream} className="stream-btn start" title="Start Stream">
              <Play size={14} />
            </button>
          ) : (
            <button onClick={stopStream} className="stream-btn stop" title="Stop Stream">
              <Square size={14} />
            </button>
          )}
        </div>
        {(hasAlert || hasViolence) && (
          <div className="alert-indicator">
            <AlertTriangle size={16} />
            <span>THREAT DETECTED</span>
          </div>
        )}
      </div>
      <div className="video-container">
        {isStreaming ? (
          <div className="real-video">
            <div className="video-info">
              <div className="video-path">{streamInfo?.video_path?.split('/').pop()}</div>
              <div className="video-stats">
                <span>FPS: {streamInfo?.fps}</span>
                <span>Frames: {streamInfo?.frame_count}</span>
                <span>Size: {streamInfo?.width}x{streamInfo?.height}</span>
              </div>
            </div>
            {videoFrame ? (
              <div className="video-frame-container">
                <img 
                  src={videoFrame} 
                  alt="Live Stream" 
                  className="video-frame"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                <div className="timestamp">{currentTime}</div>
                <div className="stream-status">LIVE STREAM</div>
              </div>
            ) : (
              <div className="video-placeholder">
                <div className="scan-line"></div>
                <div className="timestamp">{currentTime}</div>
                <div className="stream-status">LOADING...</div>
              </div>
            )}
          </div>
        ) : (
          <div className="mock-video">
            <div className="video-placeholder">
              <div className="scan-line"></div>
              <div className="timestamp">{currentTime}</div>
              <div className="stream-status">OFFLINE</div>
            </div>
          </div>
        )}
        {streamError && (
          <div className="stream-error">
            <span>{streamError}</span>
          </div>
        )}
        {(hasAlert || hasViolence) && (detection || streamInfo) && (
          <div className="threat-overlay">
            <div className="threat-box">
              <span>
                {hasViolence ? 'Violence Detected' : 
                 detection ? detection.threat_type.charAt(0).toUpperCase() + detection.threat_type.slice(1) + ' Detected' : 'Threat Detected'}
              </span>
              <span>
                Confidence: {hasViolence ? 'High' : 
                Math.round((detection?.confidence || 0) * 100)}%
              </span>
              {detection?.indicators && detection.indicators.length > 0 && (
                <span className="threat-indicators">
                  {detection.indicators.slice(0, 1).join(', ')}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraFeed;