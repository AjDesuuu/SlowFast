#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ViT-L model builder for SlowFast."""

import torch
import torch.nn as nn
from .build import MODEL_REGISTRY
from . import head_helper

@MODEL_REGISTRY.register()
class ViTL(nn.Module):
    """
    ViT-L model wrapper for SlowFast framework
    This allows using standard ViT-L checkpoints with SlowFast's AVA evaluation pipeline
    """
    
    def __init__(self, cfg):
        super(ViTL, self).__init__()
        
        # Import your existing ViT-L implementation
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), "../../../datasets/ava-dataset-utils"))
        from vjepa_ava.models.vjepa_encoder import VJEPAEncoder
        
        self.backbone = VJEPAEncoder(
            checkpoint_path=cfg.TRAIN.CHECKPOINT_FILE_PATH if hasattr(cfg.TRAIN, 'CHECKPOINT_FILE_PATH') else None,
            feature_layers=cfg.VIT.FEATURE_LAYERS if hasattr(cfg, 'VIT') else [23],  # Use last layer by default
            model_name='vit_large_patch16_224',
            img_size=cfg.DATA.TRAIN_CROP_SIZE
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.get_feature_dim()
        
        # Detection head for AVA
        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[self.feature_dim],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // 4, 1, 1]],  # Temporal pooling
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            # Classification head
            self.head = head_helper.TransformerBasicHead(
                self.feature_dim,
                cfg.MODEL.NUM_CLASSES,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
            )
    
    def forward(self, x, bboxes=None):
        # x shape: [batch, channels, temporal, height, width]
        # Reshape for ViT processing: [batch*temporal, channels, height, width]
        
        batch_size, channels, temporal, height, width = x[0].shape
        x_reshaped = x[0].view(batch_size * temporal, channels, height, width)
        
        # Extract features using ViT-L backbone
        features = self.backbone(x_reshaped)  # [batch*temporal, num_patches, feature_dim]
        
        # Reshape back: [batch, temporal, num_patches, feature_dim]
        features = features.view(batch_size, temporal, -1, self.feature_dim)
        
        # Pass through detection/classification head
        if hasattr(self, 'head'):
            if bboxes is not None:
                # Detection mode
                output = self.head([features], bboxes)
            else:
                # Classification mode - global average pooling over patches and time
                features = features.mean(dim=[1, 2])  # [batch, feature_dim]
                output = self.head(features)
        else:
            output = features
            
        return output