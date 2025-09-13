import React from 'react';
import TopBar from './TopBar';
import CameraGrid from './CameraGrid';
import Sidebar from './Sidebar';
import BottomBar from './BottomBar';
import './Dashboard.css';

const Dashboard: React.FC = () => {
  return (
    <div className="dashboard">
      <TopBar />
      <div className="main-content">
        <div className="left-section">
          <CameraGrid />
        </div>
        <div className="right-section">
          <Sidebar />
        </div>
      </div>
      <BottomBar />
    </div>
  );
};

export default Dashboard;