#!/usr/bin/env python3
"""
V-JEPA AVA Training Script
Train linear classifier on frozen V-JEPA features following paper hyperparameters
"""

import argparse
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Add SlowFast to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# SlowFast imports
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models.vjepa_ava_model import VJEPAAVAModel
from slowfast.datasets.ava_video_dataset import AvaVideoDataset
from slowfast.utils.meters import AVAMeter
from slowfast.utils.multigrid import MultigridSchedule
import slowfast.visualization.tensorboard_vis as tb

logger = logging.get_logger(__name__)


class VJEPATrainer:
    """V-JEPA AVA trainer following paper hyperparameters"""
    
    def __init__(self, cfg, use_amp=False):
        self.cfg = cfg
        self.use_amp = use_amp
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = VJEPAAVAModel(
            checkpoint_path=cfg.TRAIN.CHECKPOINT_FILE_PATH,
            num_classes=cfg.MODEL.NUM_CLASSES
        )
        
        # Only classifier parameters are trainable
        trainable_params = list(self.model.classifier.parameters())
        logger.info(f"Training {len(trainable_params)} classifier parameters")
        
        # Create optimizer following V-JEPA paper Table 10
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=cfg.SOLVER.BASE_LR,  # 1e-3 from paper
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,  # 0.05 from paper
            eps=1e-8  # opt_eps from paper
        )
        
        # Learning rate scheduler with warmup
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Loss function (multi-label BCE)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Mixed precision training
        if self.use_amp:
            try:
                self.scaler = GradScaler('cuda')
            except TypeError:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Move model to device
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = loader.construct_loader(cfg, "train")
        self.val_loader = loader.construct_loader(cfg, "val")
        
        # Create meters
        self.train_meter = AVAMeter(len(self.train_loader), cfg, mode="train")
        self.val_meter = AVAMeter(len(self.val_loader), cfg, mode="val")
        
        # Checkpointing
        self.start_epoch = 0
        self.best_mAP = 0.0
        
    def _create_lr_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(epoch):
            # epoch is 0-indexed, so we need to add 1 for proper warmup calculation
            current_epoch = epoch + 1
            if current_epoch <= self.cfg.SOLVER.WARMUP_EPOCHS:
                # Linear warmup from 0.1 to 1.0 of base LR
                warmup_factor = 0.1 + 0.9 * (current_epoch / self.cfg.SOLVER.WARMUP_EPOCHS)
                return warmup_factor
            else:
                # Cosine decay after warmup
                progress = (current_epoch - self.cfg.SOLVER.WARMUP_EPOCHS) / (self.cfg.SOLVER.MAX_EPOCH - self.cfg.SOLVER.WARMUP_EPOCHS)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        # Keep encoder frozen
        self.model.encoder.eval()
        
        self.train_meter.iter_tic()
        
        for cur_iter, (inputs, labels, idx, _, extra_data) in enumerate(self.train_loader):
            # Transfer to GPU
            if self.cfg.NUM_GPUS:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                for key, val in extra_data.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if hasattr(val[i], 'cuda'):
                                extra_data[key][i] = val[i].cuda(non_blocking=True)
                    elif hasattr(val, 'cuda'):
                        extra_data[key] = val.cuda(non_blocking=True)
            
            # Data loading is done, mark end of data time
            self.train_meter.data_toc()
            
            # Forward pass
            if self.use_amp:
                # Use mixed precision
                try:
                    # Try new PyTorch autocast API
                    autocast_context = autocast(device_type='cuda', dtype=torch.float16)
                except TypeError:
                    # Fallback to old PyTorch autocast API
                    autocast_context = autocast()
                
                with autocast_context:
                    preds = self.model(inputs, extra_data["boxes"])
                    
                    # Handle case where model returns empty predictions
                    if preds.shape[0] == 0:
                        # Skip this batch if no valid boxes
                        print("Skipping batch with no valid boxes")
                        # Skip timer updates for this iteration
                        self.train_meter.iter_tic()
                        continue
                    
                    # Compute loss
                    loss = self.criterion(preds, labels)
            else:
                # Regular forward pass without mixed precision
                preds = self.model(inputs, extra_data["boxes"])
                
                # Handle case where model returns empty predictions
                if preds.shape[0] == 0:
                    # Skip this batch if no valid boxes
                    print("Skipping batch with no valid boxes")
                    # Skip timer updates for this iteration
                    self.train_meter.iter_tic()
                    continue
                
                # Compute loss
                loss = self.criterion(preds, labels)
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                print("ERROR: Loss is NaN, skipping this batch")
                self.train_meter.iter_tic()
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp and self.scaler is not None:
                # Use gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update meters - AVA meter expects different signature
            self.train_meter.iter_toc()
            self.train_meter.update_stats(preds, extra_data["boxes"], idx, loss=loss.item(), lr=self.lr_scheduler.get_last_lr()[0])
            self.train_meter.log_iter_stats(epoch, cur_iter)
            self.train_meter.iter_tic()
        
        # Log epoch stats
        self.train_meter.log_epoch_stats(epoch)
        self.train_meter.reset()
        
        # Update learning rate
        self.lr_scheduler.step()
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        self.val_meter.iter_tic()
        
        with torch.no_grad():
            for cur_iter, (inputs, labels, idx, _, extra_data) in enumerate(self.val_loader):
                # Transfer to GPU
                if self.cfg.NUM_GPUS:
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda()
                    for key, val in extra_data.items():
                        if isinstance(val, (list,)):
                            for i in range(len(val)):
                                if hasattr(val[i], 'cuda'):
                                    extra_data[key][i] = val[i].cuda(non_blocking=True)
                        elif hasattr(val, 'cuda'):
                            extra_data[key] = val.cuda(non_blocking=True)
                
                # Forward pass
                preds = self.model(inputs, extra_data["boxes"])
                
                # Handle case where model returns empty predictions
                if preds.shape[0] == 0:
                    # Skip this batch if no valid boxes
                    continue
                
                # Update meters
                self.val_meter.iter_toc()
                self.val_meter.update_stats(
                    preds.detach().cpu(),
                    meta["ori_boxes"].detach().cpu(),
                    meta["metadata"].detach().cpu(),
                )
                self.val_meter.log_iter_stats(epoch, cur_iter)
                self.val_meter.iter_tic()
        
        # Finalize validation metrics
        self.val_meter.finalize_metrics()
        map_score = self.val_meter.full_map
        
        # Log epoch stats
        self.val_meter.log_epoch_stats(epoch)
        self.val_meter.reset()
        
        return map_score
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_mAP': self.best_mAP,
            'cfg': self.cfg,
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.cfg.OUTPUT_DIR, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.cfg.OUTPUT_DIR, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting V-JEPA AVA training")
        logger.info(f"Training for {self.cfg.SOLVER.MAX_EPOCH} epochs")
        
        for epoch in range(self.start_epoch, self.cfg.SOLVER.MAX_EPOCH):
            logger.info(f"Epoch {epoch+1}/{self.cfg.SOLVER.MAX_EPOCH}")
            
            # Train
            self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.cfg.TRAIN.EVAL_PERIOD == 0:
                map_score = self.validate(epoch)
                
                # Save checkpoint
                is_best = map_score > self.best_mAP
                if is_best:
                    self.best_mAP = map_score
                    logger.info(f"New best mAP: {self.best_mAP:.4f}")
                
                self.save_checkpoint(epoch, is_best)
        
        logger.info(f"Training completed. Best mAP: {self.best_mAP:.4f}")


def create_config():
    """Create configuration following V-JEPA paper hyperparameters"""
    from slowfast.config.defaults import get_cfg
    
    cfg = get_cfg()
    
    # Model settings
    cfg.MODEL.ARCH = "slowfast"  # Use slowfast arch to avoid NotImplementedError
    cfg.MODEL.NUM_CLASSES = 80  # AVA has 80 action classes (not 60)
    cfg.MODEL.HEAD_ACT = "sigmoid"
    
    # Training settings - From V-JEPA Paper Table 10
    cfg.SOLVER.BASE_LR = 1e-4  # lr: 0.0001 (corrected from paper)
    cfg.SOLVER.WEIGHT_DECAY = 0.05  # weight decay: 0.05
    cfg.SOLVER.MAX_EPOCH = 30  # epochs: 30
    cfg.SOLVER.WARMUP_EPOCHS = 2  # warmup epochs: 2
    cfg.SOLVER.MOMENTUM = 0.9  # momentum: 0.9
    cfg.SOLVER.OPTIMIZING_METHOD = "adamw"  # opt: AdamW
    
    # Data settings
    cfg.TRAIN.BATCH_SIZE = 64  # batch size: 64
    cfg.TEST.BATCH_SIZE = 64
    cfg.DATA_LOADER.NUM_WORKERS = 12
    
    # Dataset settings
    cfg.TRAIN.DATASET = "ava"
    cfg.TEST.DATASET = "ava"
    cfg.DETECTION.ENABLE = True
    cfg.DETECTION.ALIGNED = True
    
    # Video processing settings
    cfg.AVA.IMG_PROC_BACKEND = "pyav"  # Use PyAV for video decoding
    cfg.DATA.DECODING_BACKEND = "pyav"
    cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]  # Video scaling
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 256
    
    # AVA paths
    cfg.AVA.VIDEO_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/videos/trainval'  # Directory with cut videos
    cfg.AVA.VIDEO_EXTENSION = 'mp4'  # Video file extension
    cfg.AVA.FRAME_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/frames'  # Keep for compatibility
    cfg.AVA.FRAME_LIST_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists'
    cfg.AVA.ANNOTATION_DIR = '/home/Aaron/datasets/ava-dataset-utils/ava/annotations'
    cfg.AVA.TRAIN_LISTS = ['train.csv']
    cfg.AVA.TEST_LISTS = ['val.csv']
    cfg.AVA.TRAIN_PREDICT_BOX_LISTS = ['ava_train_predicted_boxes.csv']
    cfg.AVA.TEST_PREDICT_BOX_LISTS = ['ava_val_predicted_boxes.csv']
    cfg.AVA.GROUNDTRUTH_FILE = 'ava_val_v2.2.csv'
    cfg.AVA.EXCLUSION_FILE = 'ava_val_excluded_timestamps_v2.2.csv'
    cfg.AVA.LABEL_MAP_FILE = 'ava_action_list_v2.2_for_activitynet_2019.pbtxt'
    
    # Checkpoint
    cfg.TRAIN.CHECKPOINT_FILE_PATH = '/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar'
    cfg.TRAIN.CHECKPOINT_PERIOD = 5
    cfg.TRAIN.EVAL_PERIOD = 5
    
    # Output
    cfg.OUTPUT_DIR = '/home/Aaron/SlowFast/vjepa_ava_training'
    
    # GPU settings
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.NUM_SHARDS = 1
    
    return cfg


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='V-JEPA AVA Training')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar',
                       help='Path to V-JEPA checkpoint')
    parser.add_argument('--output-dir', type=str,
                       default='/home/Aaron/SlowFast/vjepa_ava_training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--amp', action='store_true',
                       help='Enable automatic mixed precision training')
    parser.add_argument('--workers', type=int, default=12,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create config
    cfg = create_config()
    cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.MAX_EPOCH = args.epochs
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.DATA_LOADER.NUM_WORKERS = args.workers
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Setup logging
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("V-JEPA AVA Training")
    logger.info(f"Checkpoint: {cfg.TRAIN.CHECKPOINT_FILE_PATH}")
    logger.info(f"Output dir: {cfg.OUTPUT_DIR}")
    logger.info(f"Epochs: {cfg.SOLVER.MAX_EPOCH}")
    logger.info(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    logger.info(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    
    # Create trainer
    trainer = VJEPATrainer(cfg, use_amp=args.amp)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()