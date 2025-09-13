import React from 'react';
import { Download, Play, Clock } from 'lucide-react';
import './EvidencePanel.css';

const EvidencePanel: React.FC = () => {
  const evidenceFiles = [
    { id: 1, name: 'Fighting_Incident_10_23.mp4', timestamp: '10:23 AM', duration: '2:15', size: '45MB' },
    { id: 2, name: 'Unauthorized_Access_09_45.mp4', timestamp: '09:45 AM', duration: '1:30', size: '32MB' },
    { id: 3, name: 'Suspicious_Activity_09_12.mp4', timestamp: '09:12 AM', duration: '3:45', size: '78MB' },
    { id: 4, name: 'Motion_Detected_08_30.mp4', timestamp: '08:30 AM', duration: '0:45', size: '12MB' }
  ];

  return (
    <div className="evidence-panel">
      <div className="panel-header">
        <Download size={18} />
        <span>Evidence Files</span>
      </div>
      <div className="evidence-list">
        {evidenceFiles.map(file => (
          <div key={file.id} className="evidence-item">
            <div className="file-info">
              <div className="file-name">{file.name}</div>
              <div className="file-details">
                <div className="file-timestamp">
                  <Clock size={10} />
                  <span>{file.timestamp}</span>
                </div>
                <div className="file-meta">
                  {file.duration} • {file.size}
                </div>
              </div>
            </div>
            <div className="file-actions">
              <button className="action-btn play-btn">
                <Play size={12} />
              </button>
              <button className="action-btn download-btn">
                <Download size={12} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EvidencePanel;