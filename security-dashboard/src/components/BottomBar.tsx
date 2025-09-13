import React from 'react';
import { Play, Square, Download, BarChart3 } from 'lucide-react';
import './BottomBar.css';

const BottomBar: React.FC = () => {
  return (
    <div className="bottom-bar">
      <div className="system-controls">
        <button className="control-btn start-btn">
          <Play size={16} />
          <span>Start Monitoring</span>
        </button>
        <button className="control-btn stop-btn">
          <Square size={16} />
          <span>Stop Monitoring</span>
        </button>
        <button className="control-btn export-btn">
          <Download size={16} />
          <span>Export Report</span>
        </button>
      </div>
      
      <div className="statistics">
        <div className="stat-item">
          <BarChart3 size={16} />
          <span>Incidents Today: 3</span>
        </div>
        <div className="stat-item">
          <span>Avg Response Time: 2.3 min</span>
        </div>
        <div className="stat-item">
          <span>System Load: 23%</span>
        </div>
      </div>
    </div>
  );
};

export default BottomBar;