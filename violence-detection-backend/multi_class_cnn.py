import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
from typing import List, Tuple, Dict
import random

class MultiClassViolenceDataset(Dataset):
    """Custom dataset for multi-class violence detection"""
    
    def __init__(self, frame_paths: List[str], labels: List[int], transform=None):
        self.frame_paths = frame_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        # load frame
        frame_path = self.frame_paths[idx]
        try:
            image = Image.open(frame_path).convert('RGB')
        except:
            # create dummy image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

class MultiClassViolenceCNN(nn.Module):
    """Enhanced CNN for multi-class violence classification"""
    
    def __init__(self, num_classes=13):
        super(MultiClassViolenceCNN, self).__init__()
        
        # feature extraction layers (deeper network)
        self.features = nn.Sequential(
            # first conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # fifth conv block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MultiClassTrainer:
    """Trainer class for multi-class violence detection"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
    
    def train_epoch(self, dataloader):
        """train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=30):
        """train the model for multiple epochs"""
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 20)
            
            # train
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # validate
            val_loss, val_acc = self.validate(val_loader)
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'multi_class_violence_model_best.pth')
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
            
            self.scheduler.step()
        
        return best_val_acc

def extract_frames_for_multi_class_training(data_dir: str, max_frames_per_video: int = 15) -> Tuple[List[str], List[int], Dict]:
    """extract frames from videos for multi-class training"""
    frame_paths = []
    labels = []
    
    # create frames directory
    frames_dir = os.path.join(data_dir, 'multi_class_training_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # define category mapping
    category_mapping = {
        'Abuse': 0,
        'Arrest': 1, 
        'Arson': 2,
        'Assault': 3,
        'Burglary': 4,
        'Explosion': 5,
        'Fighting': 6,
        'RoadAccidents': 7,
        'Robbery': 8,
        'Shooting': 9,
        'Shoplifting': 10,
        'Stealing': 11,
        'Vandalism': 12
    }
    
    # process each category
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) 
                  and d not in ['Labels', 'DCSASS Dataset', 'training_frames', 'multi_class_training_frames']]
    
    print(f"Found {len(categories)} categories: {categories}")
    
    for category in categories:
        if category not in category_mapping:
            print(f"⚠️ Skipping unknown category: {category}")
            continue
            
        category_path = os.path.join(data_dir, category)
        video_dirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        
        print(f"Processing {category} with {len(video_dirs)} videos...")
        
        # limit videos per category to balance dataset
        max_videos = min(len(video_dirs), 30)  # max 30 videos per category
        
        for video_dir in video_dirs[:max_videos]:
            video_path = os.path.join(category_path, video_dir)
            
            # get label
            label = category_mapping[category]
            
            # extract frames
            cap = cv2.VideoCapture(os.path.join(video_path, os.listdir(video_path)[0]))
            frame_count = 0
            
            while frame_count < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # save frame
                frame_filename = f"{category}_{video_dir}_frame_{frame_count}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                frame_paths.append(frame_path)
                labels.append(label)
                frame_count += 1
            
            cap.release()
    
    print(f"Extracted {len(frame_paths)} frames total")
    
    # print class distribution
    class_counts = {}
    for label in labels:
        class_name = list(category_mapping.keys())[list(category_mapping.values()).index(label)]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} frames")
    
    return frame_paths, labels, category_mapping

def train_multi_class_model(data_dir: str = "./data", epochs: int = 20):
    """main training function for multi-class model"""
    print("🚀 Starting multi-class violence detection model training...")
    
    # check if we have extracted frames
    frames_dir = os.path.join(data_dir, 'multi_class_training_frames')
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        print("📸 Extracting frames from videos...")
        frame_paths, labels, category_mapping = extract_frames_for_multi_class_training(data_dir)
    else:
        print("📸 Loading existing frames...")
        frame_paths = []
        labels = []
        
        # define category mapping
        category_mapping = {
            'Abuse': 0, 'Arrest': 1, 'Arson': 2, 'Assault': 3, 'Burglary': 4,
            'Explosion': 5, 'Fighting': 6, 'RoadAccidents': 7, 'Robbery': 8,
            'Shooting': 9, 'Shoplifting': 10, 'Stealing': 11, 'Vandalism': 12
        }
        
        for filename in os.listdir(frames_dir):
            if filename.endswith('.jpg'):
                frame_paths.append(os.path.join(frames_dir, filename))
                # determine label from filename
                for category, label in category_mapping.items():
                    if category in filename:
                        labels.append(label)
                        break
    
    if len(frame_paths) == 0:
        print("❌ No frames found! Please check your data directory.")
        return None, None
    
    # split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        frame_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Training samples: {len(train_paths)}")
    print(f"📊 Validation samples: {len(val_paths)}")
    
    # data transforms with more augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # create datasets
    train_dataset = MultiClassViolenceDataset(train_paths, train_labels, train_transform)
    val_dataset = MultiClassViolenceDataset(val_paths, val_labels, val_transform)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # create model
    model = MultiClassViolenceCNN(num_classes=13)
    print(f"📱 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # train model
    trainer = MultiClassTrainer(model, device)
    best_acc = trainer.train(train_loader, val_loader, epochs)
    
    print(f"🎯 Training completed! Best validation accuracy: {best_acc:.2f}%")
    
    # save category mapping
    with open('category_mapping.json', 'w') as f:
        json.dump(category_mapping, f, indent=2)
    
    return model, category_mapping

def load_multi_class_model(model_path: str):
    """Load a trained multi-class model"""
    # load category mapping
    mapping_path = 'category_mapping.json'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            category_mapping = json.load(f)
    else:
        # default mapping if file doesn't exist
        category_mapping = {
            '0': 'RoadAccidents', '1': 'Arson', '2': 'Shoplifting', '3': 'Stealing',
            '4': 'Burglary', '5': 'Fighting', '6': 'Vandalism', '7': 'Explosion',
            '8': 'Arrest', '9': 'Abuse', '10': 'Robbery', '11': 'Assault', '12': 'Shooting'
        }
    
    # create model
    model = MultiClassViolenceCNN(num_classes=len(category_mapping))
    
    # load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found")
        return None, None
    
    return model, category_mapping

if __name__ == "__main__":
    # train the multi-class model
    model, mapping = train_multi_class_model(epochs=15)  # reduced epochs for demo