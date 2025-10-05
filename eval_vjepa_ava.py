#!/usr/bin/env python3
"""
V-JEPA AVA Evaluation Script
Evaluate trained V-JEPA classifier on AVA validation set
"""

import argparse
import os
import sys
import torch
import logging

# Add SlowFast to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models.vjepa_ava_model import VJEPAAVAModel
from slowfast.utils.meters import AVAMeter

logger = logging.get_logger(__name__)


def create_config():
    """Create configuration for evaluation"""
    from slowfast.config.defaults import get_cfg
    
    cfg = get_cfg()
    
    # Model settings
    cfg.MODEL.ARCH = "slowfast"  # Use slowfast arch to avoid NotImplementedError
    cfg.MODEL.NUM_CLASSES = 80  # AVA has 80 action classes total
    cfg.MODEL.HEAD_ACT = "sigmoid"
    
    # Test settings
    cfg.TEST.ENABLE = True
    cfg.TEST.DATASET = "ava"
    cfg.TEST.BATCH_SIZE = 64
    cfg.DATA_LOADER.NUM_WORKERS = 8
    
    # Detection settings
    cfg.DETECTION.ENABLE = True
    cfg.DETECTION.ALIGNED = True
    cfg.DETECTION.SPATIAL_SCALE_FACTOR = 16
    cfg.DETECTION.ROI_XFORM_RESOLUTION = 7
    
    # AVA settings
    cfg.AVA.FRAME_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/frames'
    cfg.AVA.FRAME_LIST_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists'
    cfg.AVA.ANNOTATION_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/annotations'
    cfg.AVA.TEST_LISTS = ['val.csv']
    cfg.AVA.TEST_PREDICT_BOX_LISTS = ['ava_val_predicted_boxes.csv']
    cfg.AVA.GROUNDTRUTH_FILE = 'ava_val_v2.2.csv'
    cfg.AVA.EXCLUSION_FILE = 'ava_val_excluded_timestamps_v2.2.csv'
    cfg.AVA.LABEL_MAP_FILE = 'ava_action_list_v2.2_for_activitynet_2019.pbtxt'
    
    # GPU settings
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.NUM_SHARDS = 1
    
    return cfg


def load_trained_model(vjepa_checkpoint_path, trained_checkpoint_path=None):
    """Load V-JEPA model with trained classifier"""
    # Create model
    model = VJEPAAVAModel(
        checkpoint_path=vjepa_checkpoint_path,
        num_classes=80
    )
    
    # Load trained classifier weights if available
    if trained_checkpoint_path and os.path.exists(trained_checkpoint_path):
        logger.info(f"Loading trained classifier from {trained_checkpoint_path}")
        checkpoint = torch.load(trained_checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        logger.info("Loaded trained classifier weights")
    else:
        logger.warning("No trained classifier weights found - using random initialization")
    
    return model


def perform_test(test_loader, model, test_meter, cfg):
    """Perform evaluation"""
    model.eval()
    test_meter.iter_tic()

    with torch.no_grad():
        for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(test_loader):
            try:
                # Transfer to GPU
                if cfg.NUM_GPUS:
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
                                if hasattr(val[i], 'cuda'):
                                    meta[key][i] = val[i].cuda(non_blocking=True)
                        elif hasattr(val, 'cuda'):
                            meta[key] = val.cuda(non_blocking=True)

                # Forward pass
                preds = model(inputs, meta["boxes"])

                # Gather predictions if multi-GPU
                if cfg.NUM_GPUS > 1:
                    preds, labels, video_idx, meta = du.all_gather(
                        [preds, labels, video_idx, meta]
                    )

                test_meter.iter_toc()
                
                # Update stats
                test_meter.update_stats(
                    preds.detach().cpu(),
                    meta["ori_boxes"].detach().cpu(),
                    meta["metadata"].detach().cpu(),
                )
                test_meter.log_iter_stats(None, cur_iter)
                test_meter.iter_tic()
                
            except Exception as e:
                logger.error(f"Error in iteration {cur_iter}: {e}")
                continue

    # Finalize metrics
    test_meter.finalize_metrics()
    return test_meter


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='V-JEPA AVA Evaluation')
    parser.add_argument('--vjepa-checkpoint', type=str,
                       default='/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar',
                       help='Path to V-JEPA pretrained checkpoint')
    parser.add_argument('--trained-checkpoint', type=str,
                       default=None,
                       help='Path to trained classifier checkpoint')
    parser.add_argument('--output-dir', type=str,
                       default='/home/Aaron/SlowFast/vjepa_ava_eval_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create config
    cfg = create_config()
    cfg.OUTPUT_DIR = args.output_dir
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Setup logging
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("V-JEPA AVA Evaluation")
    logger.info(f"V-JEPA checkpoint: {args.vjepa_checkpoint}")
    logger.info(f"Trained checkpoint: {args.trained_checkpoint}")
    logger.info(f"Output dir: {cfg.OUTPUT_DIR}")
    
    # Load model
    model = load_trained_model(args.vjepa_checkpoint, args.trained_checkpoint)
    
    if cfg.NUM_GPUS:
        model = model.cuda()
    
    # Create test loader
    test_loader = loader.construct_loader(cfg, "test")
    logger.info(f"Test loader created with {len(test_loader)} batches")
    
    # Create test meter
    test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    
    # Perform evaluation
    logger.info("Starting evaluation...")
    test_meter = perform_test(test_loader, model, test_meter, cfg)
    
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to {cfg.OUTPUT_DIR}")
    
    # Print final mAP
    if hasattr(test_meter, 'full_map'):
        logger.info(f"Final mAP: {test_meter.full_map:.4f}")


if __name__ == "__main__":
    main()