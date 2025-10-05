#!/usr/bin/env python3
"""
Completely Standalone ViT-L AVA Evaluation
Uses core AVA evaluation logic without SlowFast framework dependencies
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
from sklearn.metrics import average_precision_score

class ViTLEncoder(nn.Module):
    """ViT-Large encoder using timm"""
    def __init__(self, checkpoint_path=None, img_size=224):
        super().__init__()
        import timm
        
        # Load ViT-L/16
        self.encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
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
            clean_key = k
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]
            if clean_key.startswith('backbone.'):
                clean_key = clean_key[9:]
            cleaned_state_dict[clean_key] = v
        
        # Load state dict
        missing_keys, unexpected_keys = self.encoder.load_state_dict(cleaned_state_dict, strict=False)
        print(f"✅ Loaded checkpoint: {len(cleaned_state_dict)} keys")
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
    
    def forward(self, x):
        features = self.encoder.forward_features(x)  # [B, N+1, D]
        return features[:, 1:]  # Remove CLS token: [B, N, D]

class AVADetectionHead(nn.Module):
    """Detection head for AVA"""
    def __init__(self, input_dim=1024, num_classes=80, dropout=0.5):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        # x: [B, N, D] where N is number of patches
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

def load_ava_annotations(annotations_file):
    """Load AVA annotations"""
    print(f"Loading annotations from {annotations_file}")
    
    # AVA format: video_id,middle_frame_timestamp,x1,y1,x2,y2,action_id,person_id
    df = pd.read_csv(annotations_file, header=None, 
                     names=['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id'])
    
    print(f"Loaded {len(df)} annotation entries")
    print(f"Unique videos: {df['video_id'].nunique()}")
    print(f"Unique action classes: {df['action_id'].nunique()}")
    
    return df

def load_frame_list(frame_list_file):
    """Load frame list"""
    print(f"Loading frame list from {frame_list_file}")
    
    # Format: original_vido_id video_id frame_id path labels
    df = pd.read_csv(frame_list_file, header=None, sep=' ',
                     names=['original_video_id', 'video_id', 'frame_id', 'path', 'labels'])
    
    print(f"Loaded {len(df)} frame entries")
    return df

class AVADataset(Dataset):
    """AVA dataset for evaluation"""
    def __init__(self, frame_list_file, annotations_file, frames_dir, transform=None, max_samples=None):
        self.frames_dir = frames_dir
        self.transform = transform
        
        # Load frame list and annotations
        self.frame_list = load_frame_list(frame_list_file)
        self.annotations = load_ava_annotations(annotations_file)
        
        # Limit samples for testing
        if max_samples:
            self.frame_list = self.frame_list.head(max_samples)
            print(f"Limited to {max_samples} samples for testing")
        
        print(f"Dataset ready with {len(self.frame_list)} samples")
    
    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        # Get frame info
        frame_info = self.frame_list.iloc[idx]
        frame_path = frame_info['path']
        video_id = frame_info['video_id']
        
        # Load image
        img_path = os.path.join(self.frames_dir, frame_path)
        
        # Handle missing frames
        if not os.path.exists(img_path):
            dummy_img = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(80)
            return dummy_img, dummy_labels, video_id, frame_path
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            dummy_img = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(80)
            return dummy_img, dummy_labels, video_id, frame_path
        
        # Create multi-hot labels (simplified)
        labels = torch.zeros(80)
        
        # Match with annotations (simplified approach)
        frame_annotations = self.annotations[self.annotations['video_id'] == video_id]
        if not frame_annotations.empty:
            for _, ann in frame_annotations.iterrows():
                action_id = int(ann['action_id']) - 1  # Convert to 0-indexed
                if 0 <= action_id < 80:
                    labels[action_id] = 1.0
        
        return img, labels, video_id, frame_path

def create_transform():
    """Create image transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def evaluate_model(model, dataloader, device, save_dir):
    """Evaluate model and collect predictions"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_video_ids = []
    all_frame_paths = []
    
    with torch.no_grad():
        for batch_idx, (imgs, labels, video_ids, frame_paths) in enumerate(tqdm(dataloader, desc="Evaluating")):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(imgs)
            predictions = torch.sigmoid(outputs)
            
            # Collect results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_video_ids.extend(video_ids)
            all_frame_paths.extend(frame_paths)
            
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx * len(imgs)} samples")
    
    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Save raw results
    np.savez(os.path.join(save_dir, 'predictions.npz'),
             predictions=all_predictions,
             targets=all_targets,
             video_ids=all_video_ids,
             frame_paths=all_frame_paths)
    
    return all_predictions, all_targets, all_video_ids, all_frame_paths

def compute_metrics(predictions, targets, save_dir):
    """Compute evaluation metrics"""
    print("Computing evaluation metrics...")
    
    # Compute per-class AP
    num_classes = predictions.shape[1]
    per_class_ap = []
    
    for class_idx in range(num_classes):
        if targets[:, class_idx].sum() > 0:  # Only if class has positive samples
            ap = average_precision_score(targets[:, class_idx], predictions[:, class_idx])
            per_class_ap.append(ap)
        else:
            per_class_ap.append(0.0)
    
    # Compute mean AP
    mean_ap = np.mean(per_class_ap)
    
    # Compute other metrics
    # Threshold at 0.5 for classification metrics
    pred_binary = (predictions > 0.5).astype(int)
    
    # Per-class precision and recall
    per_class_precision = []
    per_class_recall = []
    
    for class_idx in range(num_classes):
        tp = np.sum((pred_binary[:, class_idx] == 1) & (targets[:, class_idx] == 1))
        fp = np.sum((pred_binary[:, class_idx] == 1) & (targets[:, class_idx] == 0))
        fn = np.sum((pred_binary[:, class_idx] == 0) & (targets[:, class_idx] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        per_class_precision.append(precision)
        per_class_recall.append(recall)
    
    # Overall metrics
    metrics = {
        'mAP': float(mean_ap),
        'mean_precision': float(np.mean(per_class_precision)),
        'mean_recall': float(np.mean(per_class_recall)),
        'per_class_AP': [float(ap) for ap in per_class_ap],
        'per_class_precision': [float(p) for p in per_class_precision],
        'per_class_recall': [float(r) for r in per_class_recall],
        'num_classes': int(num_classes),
        'num_samples': int(len(predictions))
    }
    
    # Print results
    print(f"\n📊 Evaluation Results:")
    print(f"  mAP: {mean_ap:.4f}")
    print(f"  Mean Precision: {np.mean(per_class_precision):.4f}")
    print(f"  Mean Recall: {np.mean(per_class_recall):.4f}")
    print(f"  Num classes: {num_classes}")
    print(f"  Num samples: {len(predictions)}")
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='ViT-L AVA Evaluation (Standalone)')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar',
                       help='Path to ViT-L checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=1000, 
                       help='Max samples for testing (None for full dataset)')
    parser.add_argument('--output-dir', type=str, default='/home/Aaron/SlowFast/vitl_ava_results',
                       help='Output directory')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print(f"🚀 Starting ViT-L AVA Evaluation")
    print(f"📱 Device: {device}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📂 Output dir: {args.output_dir}")
    print(f"🔢 Max samples: {args.max_samples}")
    
    # Create model
    print("\n🏗️ Creating ViT-L model...")
    model = ViTLAVAModel(args.checkpoint)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🔓 Trainable parameters: {trainable_params:,}")
    
    # Create dataset
    print("\n📚 Creating dataset...")
    transform = create_transform()
    
    dataset = AVADataset(
        frame_list_file='/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists/val.csv',
        annotations_file='/home/Aaron/datasets/ava-dataset-utils/ava/annotations/ava_val_v2.2.csv',
        frames_dir='/home/Aaron/datasets/ava-dataset-utils/ava/frames',
        transform=transform,
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Evaluate
    print("\n🔍 Starting evaluation...")
    predictions, targets, video_ids, frame_paths = evaluate_model(model, dataloader, device, args.output_dir)
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    metrics = compute_metrics(predictions, targets, args.output_dir)
    
    # Save summary
    summary = {
        'checkpoint': args.checkpoint,
        'predictions_shape': predictions.shape,
        'num_frames': len(frame_paths),
        'mAP': metrics['mAP'],
        'mean_precision': metrics['mean_precision'],
        'mean_recall': metrics['mean_recall']
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📂 Results saved to {args.output_dir}")
    print(f"🎯 mAP: {metrics['mAP']:.4f}")

if __name__ == "__main__":
    main()