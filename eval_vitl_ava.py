#!/usr/bin/env python3
"""
ViT-L AVA Evaluation using SlowFast's Official Pipeline
This script evaluates a ViT-L model on AVA dataset using SlowFast's official evaluation code.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging

# Add SlowFast to path
sys.path.append('/home/Aaron/SlowFast')

# Import SlowFast components
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as slowfast_logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter
from slowfast.config.defaults import get_cfg
from slowfast.utils.parser import load_config, parse_args

# ViT-L Model Implementation
class ViTLEncoder(nn.Module):
    """ViT-Large encoder using timm"""
    def __init__(self, checkpoint_path=None, img_size=224):
        super().__init__()
        import timm
        
        # Load ViT-L/16
        self.encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=checkpoint_path is None,
            num_classes=0,  # No classification head
            global_pool='',  # No global pooling
            img_size=224,   # Force 224x224
            dynamic_img_size=True  # Allow dynamic resizing
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading ViT-L checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'target_encoder' in checkpoint:
                state_dict = checkpoint['target_encoder']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            # Load with flexible key matching
            model_dict = self.encoder.state_dict()
            matched_dict = {}
            
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present
                if k.startswith('module.'):
                    k = k[7:]
                
                # Remove 'backbone.' prefix if present (for V-JEPA checkpoints)
                if k.startswith('backbone.'):
                    k = k[9:]
                    
                if k in model_dict and v.shape == model_dict[k].shape:
                    matched_dict[k] = v
                    
            self.encoder.load_state_dict(matched_dict, strict=False)
            print(f"Loaded {len(matched_dict)}/{len(model_dict)} parameters")
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # x: [B, C, H, W]
        # Ensure input is 224x224
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        features = self.encoder.forward_features(x)  # [B, N+1, D] where N=num_patches
        return features[:, 1:]  # Remove CLS token, keep patch tokens

@torch.no_grad()
class ViTLAVAModel(nn.Module):
    """ViT-L model adapted for AVA evaluation using SlowFast's detection head"""
    def __init__(self, cfg):
        super().__init__()
        
        # ViT-L backbone
        checkpoint_path = getattr(cfg, 'VITL_CHECKPOINT_PATH', None)
        self.backbone = ViTLEncoder(checkpoint_path=checkpoint_path)
        
        # Detection head (borrowed from SlowFast)
        from slowfast.models import head_helper
        
        # ViT-L has 1024 feature dimension
        feature_dim = 1024
        
        self.head = head_helper.ResNetRoIHead(
            dim_in=[feature_dim],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[[cfg.DATA.NUM_FRAMES // 4, 1, 1]],
            resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
            scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            aligned=cfg.DETECTION.ALIGNED,
        )
    
    def forward(self, inputs, bboxes):
        # Extract features using ViT-L encoder  
        x = inputs[0]  # Get the input tensor
        batch_size, channels, num_frames, height, width = x.shape
        
        # Reshape to process frames individually
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features with ViT-L
        features = self.backbone(x)  # Shape: [batch_size * num_frames, num_patches, 1024]
        
        # Use CLS token (first token) or global average pooling
        if len(features.shape) == 3:
            # If features has patch dimension, use CLS token (index 0) or global average pooling
            features = features.mean(dim=1)  # Global average pooling: [batch_size * num_frames, 1024]
        
        # Reshape back to include temporal dimension and add spatial dimensions for head
        features = features.view(batch_size, num_frames, 1024)
        # Add spatial dimensions for the head: [batch_size, 1024, num_frames, 1, 1]
        features = features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply detection head
        output = self.head([features], bboxes)
        return output

def create_config():
    """Create configuration for ViT-L AVA evaluation"""
    cfg = get_cfg()
    
    # Basic settings
    cfg.MODEL.ARCH = 'mvit'
    cfg.MODEL.MODEL_NAME = 'ViTLAVAModel'
    cfg.MODEL.NUM_CLASSES = 80
    cfg.MODEL.DROPOUT_RATE = 0.5
    cfg.MODEL.HEAD_ACT = 'relu'
    
    # Data settings
    cfg.DATA.NUM_FRAMES = 4
    cfg.DATA.SAMPLING_RATE = 16
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224
    cfg.DATA.INPUT_CHANNEL_NUM = [3]
    cfg.DATA.PATH_TO_DATA_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava'
    
    # Detection settings
    cfg.DETECTION.ENABLE = True
    cfg.DETECTION.ALIGNED = True
    cfg.DETECTION.SPATIAL_SCALE_FACTOR = 16
    cfg.DETECTION.ROI_XFORM_RESOLUTION = 7
    
    # AVA specific settings
    cfg.AVA.FRAME_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/frames'
    cfg.AVA.FRAME_LIST_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists'
    cfg.AVA.ANNOTATION_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/annotations'
    cfg.AVA.TRAIN_LISTS = ['train.csv']
    cfg.AVA.TEST_LISTS = ['val.csv']
    cfg.AVA.TRAIN_PREDICT_BOX_LISTS = ['ava_train_predicted_boxes.csv']
    cfg.AVA.TEST_PREDICT_BOX_LISTS = ['ava_val_predicted_boxes.csv']
    cfg.AVA.GROUNDTRUTH_FILE = 'ava_val_v2.2.csv'
    cfg.AVA.EXCLUSION_FILE = 'ava_val_excluded_timestamps_v2.2.csv'
    cfg.AVA.LABEL_MAP_FILE = 'ava_action_list_v2.2_for_activitynet_2019.pbtxt'
    
    # Test settings
    cfg.TEST.ENABLE = True
    cfg.TEST.DATASET = 'ava'
    cfg.TEST.BATCH_SIZE = 8
    cfg.TEST.NUM_SPATIAL_CROPS = 1
    cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
    
    # Other settings
    cfg.NUM_GPUS = 1
    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.OUTPUT_DIR = '/home/Aaron/SlowFast/vitl_ava_results'
    
    # ViT-L specific
    cfg.VITL_CHECKPOINT_PATH = '/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar'
    
    return cfg

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    Perform testing on AVA dataset.
    """
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(test_loader):
        try:
            # Transfer the data to GPU
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                video_idx = video_idx.cuda()
                time = time.cuda()
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)

            test_meter.data_toc()

            # Perform the forward pass
            preds = model(inputs, meta["boxes"])

            # Gather all the predictions across all the devices to perform ensemble
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx, meta = du.all_gather(
                    [preds, labels, video_idx, meta]
                )

            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                preds.detach().cpu(),
                meta["ori_boxes"].detach().cpu(),
                meta["metadata"].detach().cpu(),
            )
            test_meter.log_iter_stats(None, cur_iter)

            test_meter.iter_tic()
            
        except Exception as e:
            print(f"Error in iteration {cur_iter}: {e}")
            print("Continuing to next iteration...")
            continue

    # Log epoch stats and print the final testing results
    test_meter.finalize_metrics()
    return test_meter

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='ViT-L AVA Evaluation')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar',
                       help='Path to ViT-L checkpoint')
    parser.add_argument('--output-dir', type=str,
                       default='/home/Aaron/SlowFast/vitl_ava_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Setup configuration
    cfg = create_config()
    cfg.VITL_CHECKPOINT_PATH = args.checkpoint
    cfg.OUTPUT_DIR = args.output_dir
    
    # Setup logging
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    slowfast_logging.setup_logging(cfg.OUTPUT_DIR)
    logger = slowfast_logging.get_logger(__name__)
    
    logger.info(f"Starting ViT-L AVA evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # Build model
    model = ViTLAVAModel(cfg)
    
    if cfg.NUM_GPUS:
        model = model.cuda()
    
    # Create test loader using SlowFast's loader
    test_loader = loader.construct_loader(cfg, "test")
    logger.info(f"Test loader created with {len(test_loader)} batches")
    
    # Create test meter using SlowFast's AVAMeter
    test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    
    # Perform evaluation
    logger.info("Starting evaluation...")
    test_meter = perform_test(test_loader, model, test_meter, cfg)
    
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    main()