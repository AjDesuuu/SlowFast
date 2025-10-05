#!/usr/bin/env python3
"""
ViT-L Evaluation using SlowFast's Official AVA Evaluation Pipeline
Uses only SlowFast's evaluation code, not the ava-dataset-utils
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import csv
import json
from collections import defaultdict

# Add SlowFast to path
sys.path.append('/home/Aaron/SlowFast')

# Import SlowFast's official AVA evaluation
from slowfast.utils.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_labelmap,
    read_exclusions,
    get_ava_eval_data,
    write_results
)
from slowfast.utils.env import pathmgr


class ViTLAVAEvaluator:
    """ViT-L evaluator using SlowFast's official AVA evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.load_ava_metadata()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_ava_metadata(self):
        """Load AVA annotation files and metadata"""
        ann_dir = self.config['ava_annotation_dir']
        
        # Load label map
        labelmap_file = os.path.join(ann_dir, 'ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt')
        self.categories, self.class_whitelist = read_labelmap(labelmap_file)
        
        # Load exclusions
        exclusions_file = os.path.join(ann_dir, 'ava_val_excluded_timestamps_v2.1.csv')
        if os.path.exists(exclusions_file):
            self.excluded_keys = read_exclusions(exclusions_file)
        else:
            self.excluded_keys = []
            
        # Load ground truth
        gt_file = os.path.join(ann_dir, 'ava_val_v2.2.csv')
        self.groundtruth = read_csv(gt_file, self.class_whitelist, load_score=False)
        
        self.logger.info(f"Loaded {len(self.class_whitelist)} classes")
        self.logger.info(f"Loaded {len(self.groundtruth[0])} ground truth samples")
        
    def load_vitl_model(self, checkpoint_path):
        """Load ViT-L model"""
        self.logger.info(f"Loading ViT-L model from {checkpoint_path}")
        
        # Import the ViT-L model
        from vitl_model import ViTLForAVA
        
        # Create and load model
        model = ViTLForAVA(
            checkpoint_path=checkpoint_path,
            num_classes=80,
            freeze_backbone=True
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        self.logger.info(f"ViT-L model loaded and moved to {device}")
        return model, device
        
    def load_ava_data(self):
        """Load AVA validation data"""
        from ava_dataloader import create_ava_dataloader
        
        # Default paths - update these to match your setup
        frame_dir = self.config.get('frame_dir', '/path/to/ava/frames')
        annotation_file = os.path.join(self.config['ava_annotation_dir'], 'ava_val_v2.2.csv')
        frame_list_file = self.config.get('frame_list_file', '/path/to/ava/frame_lists/val.csv')
        
        # Create dataloader
        dataloader, dataset = create_ava_dataloader(
            frame_dir=frame_dir,
            annotation_file=annotation_file,
            frame_list_file=frame_list_file,
            batch_size=32,
            max_samples=self.config.get('max_samples', 1000)  # Limit for testing
        )
        
        self.logger.info(f"Created dataloader with {len(dataset)} samples")
        return dataloader
        
    def run_inference(self, model, device, dataloader):
        """Run inference on validation data"""
        model.eval()
        
        all_predictions = []
        all_boxes = []
        all_metadata = []
        video_idx_to_name = {}
        video_idx = 0
        
        self.logger.info("Running inference...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                images = batch['image'].to(device)
                
                # Forward pass
                predictions = model(images)
                predictions = torch.sigmoid(predictions)  # Apply sigmoid for multi-label
                
                # Collect results
                batch_size = images.size(0)
                
                # Create dummy boxes (batch_idx, x1, y1, x2, y2)
                boxes = torch.zeros(batch_size, 5)
                boxes[:, 0] = torch.arange(batch_size) + batch_idx * dataloader.batch_size
                boxes[:, 1:] = batch['bbox']
                
                # Create metadata (video_idx, timestamp)
                metadata = torch.zeros(batch_size, 2)
                for i in range(batch_size):
                    video_id = batch['video_id'][i]
                    if video_id not in video_idx_to_name.values():
                        video_idx_to_name[video_idx] = video_id
                        video_idx += 1
                    
                    # Find video index
                    vid_idx = [k for k, v in video_idx_to_name.items() if v == video_id][0]
                    metadata[i, 0] = vid_idx
                    metadata[i, 1] = batch['timestamp'][i]
                
                all_predictions.append(predictions.cpu().numpy())
                all_boxes.append(boxes.numpy())
                all_metadata.append(metadata.numpy())
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        boxes = np.concatenate(all_boxes, axis=0)
        metadata = np.concatenate(all_metadata, axis=0)
        
        self.logger.info(f"Inference complete. Predictions shape: {predictions.shape}")
        
        return predictions, boxes, metadata, video_idx_to_name
        
    def run_evaluation(self, checkpoint_path):
        """Run complete evaluation using SlowFast's official pipeline"""
        
        # Load model
        model, device = self.load_vitl_model(checkpoint_path)
        
        # Load validation data  
        dataloader = self.load_ava_data()
        
        # Run inference
        predictions, boxes, metadata, video_idx_to_name = self.run_inference(model, device, dataloader)
        
        # Run SlowFast's official AVA evaluation
        self.logger.info("Running official AVA evaluation...")
        
        mAP = evaluate_ava(
            preds=predictions,
            original_boxes=boxes,
            metadata=metadata.tolist(),
            excluded_keys=self.excluded_keys,
            class_whitelist=self.class_whitelist,
            categories=self.categories,
            groundtruth=self.groundtruth,
            video_idx_to_name=video_idx_to_name,
            name="vitl_evaluation"
        )
        
        self.logger.info(f"Official AVA mAP: {mAP:.4f}")
        
        # Save results
        results = {
            'mAP': float(mAP),
            'num_samples': len(predictions),
            'num_classes': len(self.class_whitelist)
        }
        
        output_file = os.path.join(self.config['output_dir'], 'vitl_ava_results.json')
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Results saved to {output_file}")
        
        return mAP


def main():
    parser = argparse.ArgumentParser(description='ViT-L AVA Evaluation using SlowFast')
    parser.add_argument('--checkpoint', required=True, help='Path to ViT-L checkpoint')
    parser.add_argument('--ava-annotation-dir', required=True, help='Path to AVA annotations directory')
    parser.add_argument('--frame-dir', help='Path to AVA frames directory')
    parser.add_argument('--frame-list-file', help='Path to AVA frame list file (val.csv)')
    parser.add_argument('--output-dir', default='./vitl_evaluation_outputs', help='Output directory')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    config = {
        'ava_annotation_dir': args.ava_annotation_dir,
        'frame_dir': args.frame_dir,
        'frame_list_file': args.frame_list_file,
        'output_dir': args.output_dir,
        'checkpoint_path': args.checkpoint,
        'max_samples': args.max_samples
    }
    
    # Create evaluator and run evaluation
    evaluator = ViTLAVAEvaluator(config)
    mAP = evaluator.run_evaluation(args.checkpoint)
    
    print(f"\n🎯 Final Results:")
    print(f"   Official AVA mAP: {mAP:.4f}")
    print(f"   Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()