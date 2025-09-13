import React from 'react';
import AlertsPanel from './AlertsPanel';
import BuildingMap from './BuildingMap';
import EvidencePanel from './EvidencePanel';
import './Sidebar.css';

const Sidebar: React.FC = () => {
  return (
    <div className="sidebar">
      <AlertsPanel />
      <BuildingMap />
      <EvidencePanel />
    </div>
  );
};

export default Sidebar;