#!/usr/bin/env python3
"""
Test video-based AVA dataset loading
"""

import sys
import os
sys.path.append('/home/Aaron/SlowFast')

from slowfast.config.defaults import get_cfg
from slowfast.datasets.ava_video_dataset import AvaVideoDataset

def test_video_dataset():
    """Test video dataset loading"""
    
    # Create config
    cfg = get_cfg()
    
    # Basic settings
    cfg.MODEL.ARCH = "slowfast"
    cfg.MODEL.NUM_CLASSES = 80
    
    # Video settings
    cfg.AVA.VIDEO_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/videos/trainval'
    cfg.AVA.VIDEO_EXTENSION = 'mp4'
    cfg.AVA.FRAME_LIST_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists'
    cfg.AVA.ANNOTATION_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/annotations'
    
    # Dataset settings
    cfg.TRAIN.DATASET = "ava"
    cfg.DETECTION.ENABLE = True
    cfg.DETECTION.ALIGNED = True
    
    # Video processing
    cfg.AVA.IMG_PROC_BACKEND = "pyav"
    cfg.DATA.DECODING_BACKEND = "pyav"
    cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 256
    
    # AVA specific
    cfg.AVA.TRAIN_LISTS = ['train.csv']
    cfg.AVA.TEST_LISTS = ['val.csv'] 
    cfg.AVA.TRAIN_PREDICT_BOX_LISTS = ['ava_train_predicted_boxes.csv']
    cfg.AVA.TEST_PREDICT_BOX_LISTS = ['ava_val_predicted_boxes.csv']
    cfg.AVA.GROUNDTRUTH_FILE = 'ava_val_v2.2.csv'
    cfg.AVA.EXCLUSION_FILE = 'ava_val_excluded_timestamps_v2.2.csv'
    cfg.AVA.LABEL_MAP_FILE = 'ava_action_list_v2.2_for_activitynet_2019.pbtxt'
    
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    
    # Test directory structure
    print("🔍 Checking video directory structure:")
    video_dir = cfg.AVA.VIDEO_DIR
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"✅ Video directory exists: {video_dir}")
        print(f"📹 Found {len(videos)} MP4 videos")
        if videos:
            print(f"   Example: {videos[0]}")
    else:
        print(f"❌ Video directory not found: {video_dir}")
        return
    
    # Test dataset creation
    try:
        print("\n🔄 Creating AVA video dataset...")
        dataset = AvaVideoDataset(cfg, "train")
        print(f"✅ Dataset created successfully!")
        print(f"   Dataset size: {len(dataset)}")
        
        # Test loading one sample
        print("\n🎯 Testing sample loading...")
        sample = dataset[0]
        frames, labels, idx, _, extra_data = sample
        
        print(f"✅ Sample loaded successfully!")
        print(f"   Frames shape: {frames.shape if hasattr(frames, 'shape') else type(frames)}")
        print(f"   Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
        print(f"   Boxes shape: {extra_data['boxes'].shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_dataset()