#!/usr/bin/env python3
"""
Simple AVA Data Loader for ViT-L Evaluation
Uses minimal dependencies and integrates with SlowFast evaluation
"""

import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SimpleAVADataset(Dataset):
    """
    Simple AVA dataset for evaluation
    Loads frames and annotations for ViT-L evaluation
    """
    
    def __init__(
        self,
        frame_dir: str,
        annotation_file: str,
        frame_list_file: str,
        transform=None,
        max_samples: int = None
    ):
        self.frame_dir = frame_dir
        self.transform = transform
        self.samples = []
        
        # Load frame lists and annotations
        self.load_data(annotation_file, frame_list_file, max_samples)
        
        logger.info(f"Loaded {len(self.samples)} AVA validation samples")
        
    def load_data(self, annotation_file, frame_list_file, max_samples):
        """Load AVA annotations and frame lists"""
        
        # Load frame lists
        frame_data = {}
        if os.path.exists(frame_list_file):
            with open(frame_list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        video_id, timestamp, frame_path = parts[0], int(parts[1]), parts[2]
                        frame_data[(video_id, timestamp)] = frame_path
        
        # Load annotations
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if max_samples and i >= max_samples:
                        break
                        
                    if len(row) >= 7:
                        video_id = row[0]
                        timestamp = int(float(row[1]))
                        bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                        action_id = int(row[6])
                        
                        # Find corresponding frame
                        frame_key = (video_id, timestamp)
                        if frame_key in frame_data:
                            frame_path = frame_data[frame_key]
                        else:
                            # Construct frame path
                            frame_path = os.path.join(self.frame_dir, video_id, f"{video_id}_{timestamp:06d}.jpg")
                        
                        # Create one-hot label
                        label = np.zeros(80, dtype=np.float32)
                        if 1 <= action_id <= 80:
                            label[action_id - 1] = 1.0
                        
                        self.samples.append({
                            'video_id': video_id,
                            'timestamp': timestamp,
                            'frame_path': frame_path,
                            'bbox': bbox,
                            'label': label,
                            'action_id': action_id
                        })
        
        # If no annotation file, create dummy data for testing
        if not self.samples:
            logger.warning("No samples loaded, creating dummy data for testing")
            for i in range(100):
                label = np.zeros(80, dtype=np.float32)
                label[i % 80] = 1.0
                
                self.samples.append({
                    'video_id': f'test_video_{i:03d}',
                    'timestamp': i,
                    'frame_path': 'dummy.jpg',
                    'bbox': [0.1, 0.1, 0.9, 0.9],
                    'label': label,
                    'action_id': (i % 80) + 1
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        frame_path = sample['frame_path']
        if os.path.exists(frame_path):
            image = Image.open(frame_path).convert('RGB')
        else:
            # Create dummy image if file doesn't exist
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.from_numpy(sample['label']),
            'video_id': sample['video_id'],
            'timestamp': sample['timestamp'],
            'bbox': torch.tensor(sample['bbox'], dtype=torch.float32),
            'action_id': sample['action_id']
        }


def create_ava_dataloader(
    frame_dir: str,
    annotation_file: str,
    frame_list_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: int = None
):
    """Create AVA validation dataloader"""
    
    # Standard ImageNet transforms for ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    dataset = SimpleAVADataset(
        frame_dir=frame_dir,
        annotation_file=annotation_file,
        frame_list_file=frame_list_file,
        transform=transform,
        max_samples=max_samples
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, dataset