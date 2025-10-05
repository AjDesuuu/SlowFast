#!/usr/bin/env python3
"""
Modified AVA Dataset for Video-based Training (No Frame Extraction)
Based on SlowFast's AVA dataset but reads directly from videos
"""

import os
import torch
import numpy as np
from fvcore.common.file_io import PathManager

from slowfast.datasets.ava_dataset import Ava
from slowfast.datasets.decoder import decode
from slowfast.datasets.video_container import get_video_container
from slowfast.datasets import utils as utils
from slowfast.utils import logging

logger = logging.get_logger(__name__)


class AvaVideoDataset(Ava):
    """
    AVA Dataset that reads directly from videos instead of extracted frames
    """

    def __init__(self, cfg, split):
        """
        Initialize AVA video dataset
        """
        # Set video settings first
        self.video_dir = cfg.AVA.VIDEO_DIR  # Directory containing cut videos
        self.video_extension = cfg.AVA.VIDEO_EXTENSION  # Video file extension
        
        # Initialize parent class
        super().__init__(cfg, split)
        
        logger.info(f"AVA Video Dataset initialized:")
        logger.info(f"  Video directory: {self.video_dir}")
        logger.info(f"  Video extension: {self.video_extension}")
        logger.info(f"  Backend: {cfg.AVA.IMG_PROC_BACKEND}")

    def _load_data(self, cfg):
        """
        Load dataset annotations and create video-to-keyframe mapping
        """
        # Load annotations (same as parent)
        super()._load_data(cfg)
        
        # Create video path mapping
        self._video_paths = {}
        
        for video_name in self._video_idx_to_name:
            video_filename = f"{video_name}.{self.video_extension}"
            video_path = os.path.join(self.video_dir, video_filename)
            
            if PathManager.exists(video_path):
                self._video_paths[video_name] = video_path
            else:
                logger.warning(f"Video not found: {video_path}")
        
        logger.info(f"Found {len(self._video_paths)} videos out of {len(self._video_idx_to_name)} expected")

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.
        
        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for corresponding boxes for the current video.
            time index (zero): The time index is currently not supported for AVA.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tuple.
        if isinstance(idx, tuple):
            idx, self._num_yielded = idx
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                idx, short_cycle_idx = idx

        # Get video info
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        
        # Check if video exists
        if video_name not in self._video_paths:
            logger.error(f"Video {video_name} not found in video paths")
            # Return dummy data or raise exception
            raise FileNotFoundError(f"Video {video_name} not found")
        
        video_path = self._video_paths[video_name]

        # Get sequence of frame indices
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            self._video_idx_to_name[video_idx],
        )

        # Calculate temporal sampling parameters
        # AVA samples at 30 FPS, and we want frames around the keyframe timestamp
        keyframe_timestamp = sec  # seconds
        start_frame = int(keyframe_timestamp * 30) + seq[0] - center_idx
        num_frames = len(seq)
        
        # Load video and decode frames
        try:
            container = get_video_container(
                video_path,
                multi_thread_decode=False,
                backend=self.cfg.AVA.IMG_PROC_BACKEND
            )
            
            # Decode frames around the keyframe
            frames = decode(
                container=container,
                sampling_rate=[self._sample_rate],
                num_frames=[num_frames],
                clip_idx=0,  # Single clip
                num_clips_uniform=1,
                target_fps=30,
                backend=self.cfg.AVA.IMG_PROC_BACKEND,
                max_spatial_scale=self.cfg.DATA.TRAIN_JITTER_SCALES[0],
            )
            
            # Handle decoder output
            if frames is not None and len(frames) > 0:
                frames = frames[0]  # Get first (and only) clip
                
                # Convert to expected format: T H W C -> T C H W
                if frames.dim() == 4:  # T H W C
                    frames = frames.permute(0, 3, 1, 2)  # T C H W
                
            else:
                logger.error(f"Failed to decode video {video_path}")
                # Fallback: create dummy frames
                frames = torch.zeros(
                    (num_frames, 3, self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.TRAIN_CROP_SIZE)
                )
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            # Fallback: create dummy frames
            frames = torch.zeros(
                (num_frames, 3, self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.TRAIN_CROP_SIZE)
            )

        # Get boxes and labels (same as parent)
        boxes = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        ori_boxes = boxes.copy()

        # Apply preprocessing to frames and boxes
        if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
            # Preprocess images and boxes
            frames, boxes = self._images_and_boxes_preprocessing(frames, boxes=boxes)
            # T C H W -> C T H W
            frames = frames.permute(1, 0, 2, 3)
        else:
            # Apply transforms for other backends
            frames, boxes = self._images_and_boxes_preprocessing_cv2(frames, boxes=boxes)

        # Get labels
        label = self._get_labels(boxes, video_idx, sec_idx)

        frames = utils.pack_pathway_output(self.cfg, frames)
        
        # Create extra data
        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
            "metadata": torch.tensor([video_idx, sec_idx, sec, center_idx]),
        }

        return frames, label, idx, {}, extra_data

    def _images_and_boxes_preprocessing_cv2(self, frames, boxes):
        """
        Preprocessing for OpenCV backend (similar to parent class)
        """
        # Apply same preprocessing as parent but for video frames
        # This is a simplified version - you might need to adapt based on exact requirements
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(frames):
            frames_np = frames.cpu().numpy()
        else:
            frames_np = frames
            
        # Apply transforms (crop, flip, etc.)
        frames_processed = []
        for frame in frames_np:
            # Apply per-frame transforms
            frame_processed = frame  # Add your transforms here
            frames_processed.append(frame_processed)
        
        frames = torch.tensor(np.stack(frames_processed), dtype=torch.float32)
        
        return frames, boxes