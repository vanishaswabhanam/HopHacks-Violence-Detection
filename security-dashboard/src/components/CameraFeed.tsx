import React, { useState, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import './CameraFeed.css';

interface Detection {
  camera_id: string;
  threat_type: string;
  confidence: number;
  timestamp: string;
  severity: string;
  indicators: string[];
}

interface CameraFeedProps {
  name: string;
  cameraId: string;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ name, cameraId }) => {
  const [detection, setDetection] = useState<Detection | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());

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

    fetchDetection();
    const detectionInterval = setInterval(fetchDetection, 3000); // refresh every 3 seconds

    return () => {
      clearInterval(timeInterval);
      clearInterval(detectionInterval);
    };
  }, [cameraId]);

  const hasAlert = detection && detection.confidence > 0.5;

  return (
    <div className={`camera-feed ${hasAlert ? 'alert' : ''}`}>
      <div className="camera-header">
        <span className="camera-name">{name}</span>
        {hasAlert && (
          <div className="alert-indicator">
            <AlertTriangle size={16} />
            <span>THREAT DETECTED</span>
          </div>
        )}
      </div>
      <div className="video-container">
        <div className="mock-video">
          <div className="video-placeholder">
            <div className="scan-line"></div>
            <div className="timestamp">{currentTime}</div>
          </div>
        </div>
        {hasAlert && detection && (
          <div className="threat-overlay">
            <div className="threat-box">
              <span>{detection.threat_type.charAt(0).toUpperCase() + detection.threat_type.slice(1)} Detected</span>
              <span>Confidence: {Math.round(detection.confidence * 100)}%</span>
              {detection.indicators.length > 0 && (
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