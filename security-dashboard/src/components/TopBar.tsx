import React, { useState, useEffect } from 'react';
import { Camera, AlertTriangle, Clock } from 'lucide-react';
import './TopBar.css';

interface SystemStats {
  cameras_online: number;
  total_cameras: number;
  alerts_today: number;
  system_uptime: number;
}

const TopBar: React.FC = () => {
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleString());
  const [stats, setStats] = useState<SystemStats>({
    cameras_online: 4,
    total_cameras: 4,
    alerts_today: 0,
    system_uptime: 99.8
  });

  useEffect(() => {
    // update time every second
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date().toLocaleString());
    }, 1000);

    // fetch stats from backend
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/stats');
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
        // keep default values if API fails
      }
    };

    fetchStats();
    const statsInterval = setInterval(fetchStats, 10000); // refresh every 10 seconds

    return () => {
      clearInterval(timeInterval);
      clearInterval(statsInterval);
    };
  }, []);

  return (
    <div className="top-bar">
      <div className="system-status">
        <div className="status-item">
          <Camera size={16} />
          <span>Cameras: {stats.cameras_online}/{stats.total_cameras} Online</span>
        </div>
        <div className="status-item">
          <AlertTriangle size={16} />
          <span>Alerts Today: {stats.alerts_today}</span>
        </div>
        <div className="status-item">
          <Clock size={16} />
          <span>Uptime: {stats.system_uptime}%</span>
        </div>
      </div>
      <div className="datetime">
        {currentTime}
      </div>
    </div>
  );
};

export default TopBar;