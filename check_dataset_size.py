#!/usr/bin/env python3
"""
Check dataset size and iterations per epoch
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from slowfast.config.defaults import get_cfg
from slowfast.datasets import loader

def create_config():
    """Create configuration"""
    cfg = get_cfg()
    
    # Model settings
    cfg.MODEL.ARCH = "slowfast"
    cfg.MODEL.NUM_CLASSES = 80
    cfg.MODEL.HEAD_ACT = "sigmoid"
    
    # Data settings
    cfg.TRAIN.BATCH_SIZE = 32  # Current batch size
    cfg.TEST.BATCH_SIZE = 32
    cfg.DATA_LOADER.NUM_WORKERS = 8
    
    # Dataset settings
    cfg.TRAIN.DATASET = "ava"
    cfg.TEST.DATASET = "ava"
    cfg.DETECTION.ENABLE = True
    cfg.DETECTION.ALIGNED = True
    
    # AVA paths
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
    
    # GPU settings
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    
    return cfg

def main():
    cfg = create_config()
    
    print("Dataset Information:")
    print("=" * 50)
    
    # Check different batch sizes
    for batch_size in [32, 48, 64, 96, 128]:
        print(f"\nBatch Size: {batch_size}")
        cfg.TRAIN.BATCH_SIZE = batch_size
        cfg.TEST.BATCH_SIZE = batch_size
        
        try:
            train_loader = loader.construct_loader(cfg, "train")
            print(f"  Iterations per epoch: {len(train_loader)}")
            print(f"  Total training samples: {len(train_loader) * batch_size}")
            
            # Calculate time estimates
            time_per_iter = 0.14  # seconds (from your logs)
            epoch_time = len(train_loader) * time_per_iter / 60  # minutes
            print(f"  Estimated epoch time: {epoch_time:.1f} minutes")
            
        except Exception as e:
            print(f"  Error: {e}")
            break

if __name__ == "__main__":
    main()