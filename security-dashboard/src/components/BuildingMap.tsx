import React from 'react';
import { MapPin } from 'lucide-react';
import './BuildingMap.css';

const BuildingMap: React.FC = () => {
  const cameras = [
    { id: 1, name: 'Hallway A', x: 20, y: 30, status: 'normal' },
    { id: 2, name: 'Cafeteria', x: 70, y: 20, status: 'incident' },
    { id: 3, name: 'Main Entrance', x: 50, y: 80, status: 'normal' },
    { id: 4, name: 'Gymnasium', x: 80, y: 70, status: 'normal' }
  ];

  return (
    <div className="building-map">
      <div className="panel-header">
        <MapPin size={18} />
        <span>Building Map</span>
      </div>
      <div className="map-container">
        <svg viewBox="0 0 100 100" className="floor-plan">
          {/* Building outline */}
          <rect x="10" y="10" width="80" height="80" fill="none" stroke="#333" strokeWidth="1" />
          
          {/* Rooms */}
          <rect x="15" y="15" width="35" height="25" fill="none" stroke="#444" strokeWidth="0.5" />
          <rect x="55" y="15" width="30" height="25" fill="none" stroke="#444" strokeWidth="0.5" />
          <rect x="15" y="45" width="35" height="25" fill="none" stroke="#444" strokeWidth="0.5" />
          <rect x="55" y="45" width="30" height="25" fill="none" stroke="#444" strokeWidth="0.5" />
          
          {/* Camera locations */}
          {cameras.map(camera => (
            <g key={camera.id}>
              <circle 
                cx={camera.x} 
                cy={camera.y} 
                r="3" 
                fill={camera.status === 'incident' ? '#ff4444' : '#00aa44'}
                stroke="#fff"
                strokeWidth="0.5"
              />
              <text 
                x={camera.x} 
                y={camera.y - 6} 
                textAnchor="middle" 
                fontSize="6" 
                fill="#fff"
              >
                {camera.id}
              </text>
            </g>
          ))}
          
          {/* Legend */}
          <g transform="translate(5, 5)">
            <circle cx="0" cy="0" r="2" fill="#00aa44" />
            <text x="4" y="2" fontSize="5" fill="#ccc">Normal</text>
            <circle cx="0" cy="6" r="2" fill="#ff4444" />
            <text x="4" y="8" fontSize="5" fill="#ccc">Incident</text>
          </g>
        </svg>
      </div>
    </div>
  );
};

export default BuildingMap;