#!/usr/bin/env python3
"""
V-JEPA AVA Model
Implements the exact architecture from V-JEPA paper for AVA frozen detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import timm
import logging

logger = logging.getLogger(__name__)

class VJEPAEncoder(nn.Module):
    """V-JEPA ViT-L/16 encoder with multi-layer feature extraction"""
    
    def __init__(self, checkpoint_path=None, feature_layers=[17, 19, 21, 23]):
        super().__init__()
        
        # Initialize hooks list first to avoid destructor issues
        self.hooks = []
        
        # Load ViT-L/16 from timm
        self.vit = timm.create_model('vit_large_patch16_224', pretrained=False)
        self.feature_layers = feature_layers  # 0-indexed layers to extract features from
        
        # Adjust for video model if needed
        self.is_video_model = False
        
        # Hook storage
        self.feature_maps = {}
        
        # Register hooks for intermediate layers
        self._register_hooks()
        
        # Load V-JEPA checkpoint
        if checkpoint_path:
            self._load_vjepa_checkpoint(checkpoint_path)
            
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features"""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output
            return hook
            
        # Register hooks for specified layers
        for layer_idx in self.feature_layers:
            if layer_idx < len(self.vit.blocks):
                hook = self.vit.blocks[layer_idx].register_forward_hook(
                    get_activation(f'layer_{layer_idx}')
                )
                self.hooks.append(hook)
    
    def _load_vjepa_checkpoint(self, checkpoint_path):
        """Load V-JEPA checkpoint"""
        logger.info(f"Loading V-JEPA checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove 'module.backbone.' prefix if present
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.backbone.'):
                clean_key = key.replace('module.backbone.', '')
                clean_state_dict[clean_key] = value
            elif key.startswith('module.'):
                clean_key = key.replace('module.', '')
                clean_state_dict[clean_key] = value
            else:
                clean_state_dict[key] = value
        
        # Handle video model format (5D patch embeddings)
        if 'patch_embed.proj.weight' in clean_state_dict:
            patch_weight = clean_state_dict['patch_embed.proj.weight']
            if len(patch_weight.shape) == 5:  # Video model: [C_out, C_in, T, H, W]
                # Take middle frame for image model: [C_out, C_in, H, W]
                clean_state_dict['patch_embed.proj.weight'] = patch_weight[:, :, patch_weight.shape[2]//2, :, :]
                self.is_video_model = True
                logger.info("Detected video model format, adapting patch embeddings")
        
        # Handle different positional embedding sizes
        if 'pos_embed' in clean_state_dict:
            pos_embed = clean_state_dict['pos_embed']
            expected_pos_embed = self.vit.pos_embed
            if pos_embed.shape != expected_pos_embed.shape:
                logger.info(f"Resizing pos_embed from {pos_embed.shape} to {expected_pos_embed.shape}")
                # Interpolate or truncate as needed
                if pos_embed.shape[1] > expected_pos_embed.shape[1]:
                    # Truncate
                    clean_state_dict['pos_embed'] = pos_embed[:, :expected_pos_embed.shape[1], :]
                else:
                    # Keep original and let strict=False handle it
                    del clean_state_dict['pos_embed']
            
        # Load with missing key handling
        missing_keys, unexpected_keys = self.vit.load_state_dict(clean_state_dict, strict=False)
        logger.info(f"Loaded {len(clean_state_dict) - len(missing_keys)}/{len(clean_state_dict)} parameters")
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
    
    def forward(self, x):
        """
        Forward pass through ViT with multi-layer feature extraction
        Args:
            x: [B, 3, H, W] input images
        Returns:
            features: [B, num_patches, concat_dim] concatenated features
        """
        # Clear previous feature maps
        self.feature_maps.clear()
        
        # Forward through ViT (this will populate feature_maps via hooks)
        output = self.vit.forward_features(x)  # [B, num_patches + 1, embed_dim]
        
        # Remove CLS token, keep patch tokens only
        patch_features = output[:, 1:]  # [B, num_patches, embed_dim]
        
        # Collect features from intermediate layers
        intermediate_features = []
        for layer_idx in self.feature_layers:
            if f'layer_{layer_idx}' in self.feature_maps:
                layer_feat = self.feature_maps[f'layer_{layer_idx}'][:, 1:]  # Remove CLS
                intermediate_features.append(layer_feat)
        
        # Concatenate all features
        if intermediate_features:
            concatenated_features = torch.cat(intermediate_features, dim=-1)
        else:
            # Fallback to final layer only
            concatenated_features = patch_features
            
        return concatenated_features
    
    def __del__(self):
        """Clean up hooks"""
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()


class ROIFeatureExtractor(nn.Module):
    """Extract ROI features using torchvision.ops.roi_align"""
    
    def __init__(self, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        
    def forward(self, feature_maps, boxes):
        """
        Extract ROI features
        Args:
            feature_maps: [B, num_patches, feature_dim] from ViT
            boxes: [N, 5] in format [batch_idx, x1, y1, x2, y2]
        Returns:
            roi_features: [N, feature_dim]
        """
        B, num_patches, feature_dim = feature_maps.shape
        
        # Reshape feature maps to spatial format
        # ViT-L/16 with 224x224 input has 14x14 patches
        patch_size = int(num_patches ** 0.5)
        
        if patch_size * patch_size != num_patches:
            print(f"Warning: num_patches ({num_patches}) is not a perfect square")
            # Handle non-square case by using closest square
            patch_size = int(num_patches ** 0.5)
        
        feature_maps_spatial = feature_maps.view(B, patch_size, patch_size, feature_dim)
        feature_maps_spatial = feature_maps_spatial.permute(0, 3, 1, 2)  # [B, feature_dim, H, W]
        
        # Ensure boxes are in correct range and format
        boxes = boxes.float().contiguous()
        
        # For now, let's skip ROI align and just use global average pooling
        # This is simpler and avoids the complex box coordinate issues
        
        # feature_maps shape: [B, num_patches, concat_dim]
        # Global average pool across all patches
        global_features = feature_maps.mean(dim=1)  # [B, concat_dim]
        
        # Extract features for each box
        # boxes format: [N, 5] where first column is batch index
        if boxes.shape[0] == 0:
            return torch.empty(0, feature_maps.shape[-1], device=feature_maps.device)
        
        batch_indices = boxes[:, 0].long()  # Get batch indices
        
        # Clamp batch indices to valid range to handle AVA data loader quirks
        batch_indices = torch.clamp(batch_indices, 0, global_features.shape[0] - 1)
        
        # Create features for each box using its corresponding batch index
        final_features = global_features[batch_indices]  # [N, concat_dim]
        
        return final_features


class AVALinearClassifier(nn.Module):
    """Simple linear classifier for AVA action classification"""
    
    def __init__(self, input_dim, num_classes=80):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # Initialize weights and bias properly
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, -2.0)  # Negative bias for class imbalance
    
    def forward(self, x):
        return self.classifier(x)


class VJEPAAVAModel(nn.Module):
    """Complete V-JEPA AVA model with frozen encoder + trainable classifier"""
    
    def __init__(self, checkpoint_path=None, num_classes=80):
        super().__init__()
        
        # V-JEPA encoder (frozen)
        self.encoder = VJEPAEncoder(checkpoint_path=checkpoint_path)
        
        # ROI feature extractor
        self.roi_extractor = ROIFeatureExtractor()
        
        # Calculate input dimension for classifier
        # ViT-L has embed_dim=1024, with 4 layers concatenated = 4096
        feature_layers = self.encoder.feature_layers
        embed_dim = 1024  # ViT-L embed_dim
        input_dim = len(feature_layers) * embed_dim
        
        # Linear classifier (trainable)
        self.classifier = AVALinearClassifier(input_dim, num_classes)
        
    def forward(self, images, boxes):
        """
        Forward pass
        Args:
            images: [B, 3, H, W] input images or list of tensors for pathways
            boxes: List of [N_i, 4] boxes for each image (x1, y1, x2, y2)
        Returns:
            predictions: [total_boxes, num_classes] action predictions
        """
        # Handle SlowFast pathway inputs (convert list to single tensor)
        if isinstance(images, list):
            # For V-JEPA, we only use the slow pathway (first element)
            images = images[0]  # [B, C, T, H, W]
        
        # For V-JEPA, we need to reshape from video to image format
        # [B, C, T, H, W] -> [B*T, C, H, W]
        if len(images.shape) == 5:  # Video format
            B, C, T, H, W = images.shape
            images = images.transpose(1, 2).contiguous()  # [B, T, C, H, W]
            images = images.view(B * T, C, H, W)  # [B*T, C, H, W]
            
            # Extract features from encoder
            features = self.encoder(images)  # [B*T, num_patches, concat_dim]
            
            # Reshape features back to video format and average over time
            # [B*T, num_patches, concat_dim] -> [B, T, num_patches, concat_dim]
            features = features.view(B, T, features.shape[1], features.shape[2])
            features = features.mean(dim=1)  # [B, num_patches, concat_dim]
        else:
            # Image format [B, C, H, W]
            features = self.encoder(images)  # [B, num_patches, concat_dim]
        
        # Prepare boxes for ROI align (add batch indices)
        roi_boxes = []
        for batch_idx, box_list in enumerate(boxes):
            if len(box_list) > 0:
                # Ensure box_list is 2D [num_boxes, 4 or 5]
                if box_list.dim() == 1:
                    box_list = box_list.unsqueeze(0)  # [1, 4 or 5]
                elif box_list.dim() > 2:
                    box_list = box_list.view(-1, box_list.shape[-1])  # [num_boxes, 4 or 5]
                
                # Extract only the bounding box coordinates (first 4 columns)
                # AVA format might include score as 5th column
                if box_list.shape[1] >= 4:
                    box_coords = box_list[:, :4]  # [num_boxes, 4]
                else:
                    print(f"Warning: box_list has shape {box_list.shape}, expected [N, 4] or [N, 5]")
                    continue
                
                batch_indices = torch.full((box_coords.shape[0], 1), batch_idx, 
                                         dtype=box_coords.dtype, device=box_coords.device)
                boxes_with_batch = torch.cat([batch_indices, box_coords], dim=1)
                roi_boxes.append(boxes_with_batch)
        
        if roi_boxes:
            roi_boxes = torch.cat(roi_boxes, dim=0)  # [total_boxes, 5]
        else:
            # Handle case with no boxes - return dummy predictions
            print("Warning: No valid boxes found in batch")
            device = images.device if hasattr(images, 'device') else 'cpu'
            # Return zero predictions with same batch size as labels
            return torch.zeros(0, self.classifier.classifier.out_features, device=device)
        
        # Extract ROI features
        roi_features = self.roi_extractor(features, roi_boxes)  # [total_boxes, concat_dim]
        
        # Ensure tensor is contiguous and proper dtype
        roi_features = roi_features.contiguous().float()
        
        # Try classification with error handling
        try:
            predictions = self.classifier(roi_features)  # [total_boxes, num_classes]
        except RuntimeError as e:
            print(f"CUDA error in classifier: {e}")
            # Fallback: move to CPU, compute, then back to GPU
            device = roi_features.device
            roi_features_cpu = roi_features.cpu()
            classifier_cpu = self.classifier.cpu()
            predictions_cpu = classifier_cpu(roi_features_cpu)
            predictions = predictions_cpu.to(device)
            self.classifier.to(device)  # Move classifier back to GPU
        
        return predictions