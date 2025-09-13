import React from 'react';
import CameraFeed from './CameraFeed';
import './CameraGrid.css';

const CameraGrid: React.FC = () => {
  const cameras = [
    { id: 'cam_7', name: 'Hallway A' },
    { id: 'cam_8', name: 'Cafeteria' },
    { id: 'cam_9', name: 'Main Entrance' },
    { id: 'cam_10', name: 'Gymnasium' }
  ];

  return (
    <div className="camera-grid">
      {cameras.map(camera => (
        <CameraFeed 
          key={camera.id} 
          name={camera.name} 
          cameraId={camera.id}
        />
      ))}
    </div>
  );
};

export default CameraGrid;