import React, { useState, useEffect } from 'react';
import { AlertTriangle, Clock } from 'lucide-react';
import './AlertsPanel.css';

interface Detection {
  camera_id: string;
  threat_type: string;
  confidence: number;
  timestamp: string;
  location: { x: number; y: number };
  severity: string;
  indicators: string[];
  filename: string;
  frames_analyzed: number;
}

const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch('http://localhost:8000/detections');
        const data = await response.json();
        setAlerts(data.detections || []);
      } catch (error) {
        console.error('Failed to fetch alerts:', error);
        // fallback to mock data if API fails
        setAlerts([
          { camera_id: 'cam_1', threat_type: 'Fighting detected', confidence: 87, timestamp: '10:23 AM', location: { x: 0, y: 0 }, severity: 'high', indicators: [], filename: '', frames_analyzed: 0 },
          { camera_id: 'cam_2', threat_type: 'Unauthorized access', confidence: 92, timestamp: '09:45 AM', location: { x: 0, y: 0 }, severity: 'high', indicators: [], filename: '', frames_analyzed: 0 }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
    // refresh every 5 seconds
    const interval = setInterval(fetchAlerts, 5000);
    return () => clearInterval(interval);
  }, []);

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'high': return '#ff4444';
      case 'medium': return '#ffaa00';
      case 'low': return '#00aa44';
      default: return '#666';
    }
  };

  if (loading) {
    return (
      <div className="alerts-panel">
        <div className="panel-header">
          <AlertTriangle size={18} />
          <span>Live Alerts</span>
        </div>
        <div className="alerts-list">
          <div className="loading">Loading alerts...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="alerts-panel">
      <div className="panel-header">
        <AlertTriangle size={18} />
        <span>Live Alerts ({alerts.length})</span>
      </div>
      <div className="alerts-list">
        {alerts.length === 0 ? (
          <div className="no-alerts">No active alerts</div>
        ) : (
          alerts.map((alert, index) => (
            <div key={index} className="alert-item">
              <div className="alert-time">
                <Clock size={12} />
                <span>{alert.timestamp}</span>
              </div>
              <div className="alert-content">
                <div className="alert-type" style={{ color: getAlertColor(alert.severity) }}>
                  {alert.threat_type.charAt(0).toUpperCase() + alert.threat_type.slice(1)} detected
                </div>
                <div className="alert-details">
                  {alert.camera_id} • Confidence: {Math.round(alert.confidence * 100)}%
                </div>
                {alert.indicators.length > 0 && (
                  <div className="alert-indicators">
                    {alert.indicators.slice(0, 2).join(', ')}
                    {alert.indicators.length > 2 && ` +${alert.indicators.length - 2} more`}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertsPanel;