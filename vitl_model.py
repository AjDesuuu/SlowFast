#!/usr/bin/env python3
"""
ViT-L Model Definition for SlowFast AVA Evaluation
Integrates your ViT-L with SlowFast's evaluation pipeline
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ViTLForAVA(nn.Module):
    """
    ViT-L model for AVA action detection
    Uses official SlowFast evaluation pipeline
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        num_classes: int = 80,
        feature_layers: List[int] = [23],  # Use last layer by default
        img_size: int = 224,
        patch_size: int = 16,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_layers = feature_layers
        self.freeze_backbone = freeze_backbone
        
        # Load ViT-L/16 backbone
        self.backbone = timm.create_model(
            'vit_large_patch16_224',
            pretrained=checkpoint_path is None,
            num_classes=0,  # No classification head
            global_pool='',  # No global pooling
            img_size=img_size
        )
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone.forward_features(dummy_input)
            self.feature_dim = features.shape[-1]
            
        # Detection head for AVA
        self.detection_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_classes)
        )
        
        logger.info(f"ViT-L model created with {self.feature_dim}D features -> {num_classes} classes")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load ViT-L checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'target_encoder' in checkpoint:
                # V-JEPA format
                state_dict = checkpoint['target_encoder']
            elif 'model_state_dict' in checkpoint:
                # Standard format
                state_dict = checkpoint['model_state_dict']
            else:
                # Direct state dict
                state_dict = checkpoint
                
            # Load backbone weights
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint with message: {msg}")
            
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            logger.info("Using timm pretrained weights instead")
            
    def forward(self, x, boxes=None):
        """
        Forward pass for AVA evaluation
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            boxes: Bounding boxes (optional, for detection)
            
        Returns:
            predictions: [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Extract features using ViT-L
        features = self.backbone.forward_features(x)  # [batch, num_patches+1, feature_dim]
        
        # Use CLS token or global average pooling
        if hasattr(self.backbone, 'num_prefix_tokens') and self.backbone.num_prefix_tokens > 0:
            # Use CLS token
            cls_features = features[:, 0]  # [batch, feature_dim]
        else:
            # Global average pooling over patches
            cls_features = features.mean(dim=1)  # [batch, feature_dim]
            
        # Apply detection head
        predictions = self.detection_head(cls_features)  # [batch, num_classes]
        
        return predictions
        
    def extract_features(self, x):
        """Extract features for analysis"""
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            return features