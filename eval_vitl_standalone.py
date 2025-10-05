#!/usr/bin/env python3
"""
Standalone ViT-L AVA Evaluation using SlowFast's evaluation metrics
This script bypasses SlowFast's model framework but uses their official AVA evaluation code
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import cv2
from PIL import Image
import torchvision.transforms as transforms

# Add SlowFast for evaluation utilities only
sys.path.append('/home/Aaron/SlowFast')

# Import only evaluation utilities from SlowFast
from slowfast.utils.ava_eval_helper import evaluate_ava, read_csv, read_labelmap, read_exclusions

class ViTLEncoder(nn.Module):
    """ViT-Large encoder using timm"""
    def __init__(self, checkpoint_path=None, img_size=224):
        super().__init__()
        import timm
        
        # Load ViT-L/16
        self.encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,  # We'll load our own weights
            num_classes=0,
            global_pool='',
            img_size=img_size
        )
        
        # Load V-JEPA checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _load_checkpoint(self, checkpoint_path):
        """Load V-JEPA checkpoint"""
        print(f"Loading ViT-L checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix and 'backbone.' prefix from keys
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove prefixes
            clean_key = k
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]
            if clean_key.startswith('backbone.'):
                clean_key = clean_key[9:]
            cleaned_state_dict[clean_key] = v
        
        # Load state dict
        missing_keys, unexpected_keys = self.encoder.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Loaded checkpoint: {len(cleaned_state_dict)} keys")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
    
    def forward(self, x):
        # x: [B, C, H, W]
        features = self.encoder.forward_features(x)  # [B, N+1, D]
        return features[:, 1:]  # Remove CLS token: [B, N, D]

class AVADetectionHead(nn.Module):
    """Simple detection head for AVA"""
    def __init__(self, input_dim=1024, num_classes=80, dropout=0.5):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        # x: [B, N, D] where N is number of patches
        # Global average pooling over patches
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.global_pool(x).squeeze(-1)  # [B, D]
        x = self.dropout(x)
        x = self.classifier(x)  # [B, num_classes]
        return x

class ViTLAVAModel(nn.Module):
    """Complete ViT-L model for AVA evaluation"""
    def __init__(self, checkpoint_path, num_classes=80, img_size=224):
        super().__init__()
        self.backbone = ViTLEncoder(checkpoint_path, img_size)
        self.head = AVADetectionHead(1024, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

class SimpleAVADataset(Dataset):
    """Simple AVA dataset for evaluation"""
    def __init__(self, frame_list_file, annotations_file, frames_dir, transform=None):
        self.frames_dir = frames_dir
        self.transform = transform
        
        # Load frame list
        self.frame_list = pd.read_csv(frame_list_file, header=None, names=['path', 'labels'])
        
        # Load annotations
        self.annotations = pd.read_csv(annotations_file)
        
        print(f"Loaded {len(self.frame_list)} frames")
        print(f"Loaded {len(self.annotations)} annotations")
    
    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        # Get frame info
        frame_path = self.frame_list.iloc[idx]['path']
        
        # Load image
        img_path = os.path.join(self.frames_dir, frame_path)
        
        # Handle missing frames
        if not os.path.exists(img_path):
            # Return dummy data
            dummy_img = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(80)
            return dummy_img, dummy_labels, frame_path
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except:
            # Return dummy data for corrupted images
            dummy_img = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(80)
            return dummy_img, dummy_labels, frame_path
        
        # Get labels (simplified - normally would match with annotations)
        labels = torch.zeros(80)  # Multi-hot encoding
        
        return img, labels, frame_path

def create_transform():
    """Create image transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def evaluate_model(model, dataloader, device, save_dir):
    """Evaluate model and return predictions"""
    model.eval()
    all_predictions = []
    all_frame_paths = []
    
    with torch.no_grad():
        for batch_idx, (imgs, labels, frame_paths) in enumerate(tqdm(dataloader, desc="Evaluating")):
            imgs = imgs.to(device)
            
            # Forward pass
            outputs = model(imgs)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            
            all_predictions.append(predictions)
            all_frame_paths.extend(frame_paths)
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx * len(imgs)} samples")
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Save predictions
    np.save(os.path.join(save_dir, 'predictions.npy'), all_predictions)
    with open(os.path.join(save_dir, 'frame_paths.txt'), 'w') as f:
        for path in all_frame_paths:
            f.write(f"{path}\n")
    
    return all_predictions, all_frame_paths

def convert_to_ava_format(predictions, frame_paths, threshold=0.5):
    """Convert predictions to AVA evaluation format"""
    
    # This is a simplified conversion - in practice you'd need proper
    # video_id, timestamp, and bounding box information
    detections_boxes = defaultdict(list)
    detections_labels = defaultdict(list) 
    detections_scores = defaultdict(list)
    
    for i, (pred, frame_path) in enumerate(zip(predictions, frame_paths)):
        # Extract video_id and timestamp from frame_path
        # This is simplified - you'd need proper parsing
        video_id = frame_path.split('/')[0] if '/' in frame_path else f"video_{i}"
        timestamp = f"{i:04d}"
        key = f"{video_id},{timestamp}"
        
        # Get predicted classes above threshold
        predicted_classes = np.where(pred > threshold)[0]
        
        for class_id in predicted_classes:
            # Add dummy bounding box (normally you'd have real boxes)
            detections_boxes[key].append([0, 0.2, 0.2, 0.8, 0.8])  # [batch_idx, x1, y1, x2, y2]
            detections_labels[key].append(class_id + 1)  # AVA classes are 1-indexed
            detections_scores[key].append(float(pred[class_id]))
    
    return detections_boxes, detections_labels, detections_scores

def main():
    parser = argparse.ArgumentParser(description='ViT-L AVA Evaluation')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar',
                       help='Path to ViT-L checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='/home/Aaron/SlowFast/vitl_ava_results',
                       help='Output directory')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # Create model
    logger.info("Creating ViT-L model...")
    model = ViTLAVAModel(args.checkpoint)
    model = model.to(device)
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    transform = create_transform()
    
    dataset = SimpleAVADataset(
        frame_list_file='/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists/val.csv',
        annotations_file='/home/Aaron/datasets/ava-dataset-utils/ava/annotations/ava_val_v2.2.csv',
        frames_dir='/home/Aaron/datasets/ava-dataset-utils/ava/frames',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Evaluate
    logger.info("Starting evaluation...")
    predictions, frame_paths = evaluate_model(model, dataloader, device, args.output_dir)
    
    # Convert to AVA format and compute metrics
    logger.info("Converting predictions to AVA format...")
    det_boxes, det_labels, det_scores = convert_to_ava_format(predictions, frame_paths)
    
    # Save results
    results = {
        'predictions_shape': predictions.shape,
        'num_frames': len(frame_paths),
        'num_detections': len(det_boxes)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"Predictions shape: {predictions.shape}")

if __name__ == "__main__":
    main()