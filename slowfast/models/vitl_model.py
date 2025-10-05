#!/usr/bin/env python3
"""
Register ViT-L model with SlowFast and use their standard evaluation pipeline
"""

import torch
import torch.nn as nn
from slowfast.models.build import MODEL_REGISTRY
from slowfast.models import head_helper

@MODEL_REGISTRY.register()
class ViTLAVAModel(nn.Module):
    """ViT-L model for AVA evaluation using SlowFast framework"""
    
    def __init__(self, cfg):
        super().__init__()
        
        # ViT-L backbone
        self.backbone = self._build_vitl_backbone(cfg)
        
        # Detection head using SlowFast's implementation
        feature_dim = 1024  # ViT-L feature dimension
        
        self.head = head_helper.ResNetRoIHead(
            dim_in=[feature_dim],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[[4, 1, 1]],  # Use the actual number of frames (4)
            resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
            scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            aligned=cfg.DETECTION.ALIGNED,
        )
    
    def _build_vitl_backbone(self, cfg):
        """Build ViT-L backbone"""
        import timm
        
        # Load ViT-L/16
        encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=True,
            num_classes=0,
            global_pool='',
            img_size=cfg.DATA.TEST_CROP_SIZE
        )
        
        # Load custom checkpoint if provided
        if hasattr(cfg, 'VITL_CHECKPOINT_PATH') and cfg.VITL_CHECKPOINT_PATH:
            self._load_vitl_checkpoint(encoder, cfg.VITL_CHECKPOINT_PATH)
        
        # Freeze backbone
        for param in encoder.parameters():
            param.requires_grad = False
            
        return encoder
    
    def _load_vitl_checkpoint(self, model, checkpoint_path):
        """Load ViT-L checkpoint"""
        import os
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found, using timm pretrained weights")
            return
            
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
        model_dict = model.state_dict()
        matched_dict = {}
        
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith('module.'):
                k = k[7:]
                
            if k in model_dict and v.shape == model_dict[k].shape:
                matched_dict[k] = v
                
        model.load_state_dict(matched_dict, strict=False)
        print(f"Loaded {len(matched_dict)}/{len(model_dict)} parameters from checkpoint")
    
    def forward(self, x, bboxes=None):
        # x is a list with one element: [B, C, T, H, W]
        x = x[0]
        B, C, T, H, W = x.shape
        
        # Reshape to process each frame: [B*T, C, H, W]
        x_reshaped = x.view(B * T, C, H, W)
        
        # Extract features with ViT-L
        features = self.backbone.forward_features(x_reshaped)  # [B*T, N+1, 1024]
        features = features[:, 1:]  # Remove CLS token: [B*T, N, 1024]
        
        # Global average pooling over patches
        features = features.mean(dim=1)  # [B*T, 1024]
        
        # Reshape back: [B, T, 1024]
        features = features.view(B, T, -1)
        
        # Reshape for detection head: [B, 1024, T, 1, 1]
        features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        
        # Pass through detection head
        output = self.head([features], bboxes)
        return output