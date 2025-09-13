import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from typing import Tuple, Dict, List
import json

class ViolenceCNN(nn.Module):
    """Simple CNN for binary violence classification (same as training)"""
    
    def __init__(self, num_classes=2):
        super(ViolenceCNN, self).__init__()
        
        # feature extraction layers
        self.features = nn.Sequential(
            # first conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNNViolenceDetector:
    """CNN-based violence detection for real-time inference"""
    
    def __init__(self, model_path: str = "violence_model_best.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViolenceCNN(num_classes=2)
        
        # load trained model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded CNN model from {model_path}")
        else:
            print(f"⚠️ Model file {model_path} not found, using untrained model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # class names
        self.class_names = ['Normal', 'Abnormal']
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """preprocess a single frame for CNN inference"""
        # convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # apply transforms
        tensor = self.transform(pil_image)
        
        # add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """predict violence on a single frame"""
        with torch.no_grad():
            # preprocess frame
            input_tensor = self.preprocess_frame(frame).to(self.device)
            
            # get prediction
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # get confidence scores
            normal_conf = probabilities[0][0].item()
            abnormal_conf = probabilities[0][1].item()
            
            # determine prediction
            predicted_class = 1 if abnormal_conf > normal_conf else 0
            confidence = max(normal_conf, abnormal_conf)
            
            return {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'normal_confidence': normal_conf,
                'abnormal_confidence': abnormal_conf
            }
    
    def predict_frames_batch(self, frames: List[np.ndarray]) -> List[Dict[str, float]]:
        """predict violence on multiple frames"""
        results = []
        
        for frame in frames:
            try:
                result = self.predict_frame(frame)
                results.append(result)
            except Exception as e:
                print(f"Error processing frame: {e}")
                # return default result for failed frames
                results.append({
                    'predicted_class': 0,
                    'class_name': 'Normal',
                    'confidence': 0.5,
                    'normal_confidence': 0.5,
                    'abnormal_confidence': 0.5
                })
        
        return results
    
    def analyze_video_segment(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """analyze a video segment and return aggregated results"""
        if not frames:
            return {
                'status': 'error',
                'message': 'No frames provided',
                'threat_level': 'normal',
                'confidence': 0.0
            }
        
        # get predictions for all frames
        predictions = self.predict_frames_batch(frames)
        
        # calculate statistics
        abnormal_count = sum(1 for p in predictions if p['predicted_class'] == 1)
        total_frames = len(predictions)
        abnormal_ratio = abnormal_count / total_frames
        
        # calculate average confidence
        avg_confidence = sum(p['confidence'] for p in predictions) / total_frames
        avg_abnormal_conf = sum(p['abnormal_confidence'] for p in predictions) / total_frames
        
        # determine threat level
        if abnormal_ratio > 0.7 and avg_abnormal_conf > 0.6:
            threat_level = 'high'
            threat_type = 'violence_detected'
        elif abnormal_ratio > 0.4 and avg_abnormal_conf > 0.5:
            threat_level = 'medium'
            threat_type = 'suspicious_activity'
        else:
            threat_level = 'low'
            threat_type = 'normal'
        
        return {
            'status': 'success',
            'threat_level': threat_level,
            'threat_type': threat_type,
            'confidence': avg_confidence,
            'abnormal_ratio': abnormal_ratio,
            'abnormal_frames': abnormal_count,
            'total_frames': total_frames,
            'avg_abnormal_confidence': avg_abnormal_conf,
            'frame_predictions': predictions[:5]  # include first 5 frame predictions for debugging
        }

def test_cnn_detector():
    """test the CNN detector with sample frames"""
    print("🧪 Testing CNN Violence Detector...")
    
    detector = CNNViolenceDetector()
    
    # create a test frame (random noise)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # test single frame prediction
    result = detector.predict_frame(test_frame)
    print(f"Single frame prediction: {result}")
    
    # test batch prediction
    test_frames = [test_frame, test_frame, test_frame]
    batch_results = detector.predict_frames_batch(test_frames)
    print(f"Batch predictions: {len(batch_results)} results")
    
    # test video segment analysis
    segment_result = detector.analyze_video_segment(test_frames)
    print(f"Video segment analysis: {segment_result}")
    
    print("✅ CNN detector test completed!")

if __name__ == "__main__":
    test_cnn_detector()