#!/bin/bash
"""
Download AVA annotations for evaluation
Run this first to get the required annotation files
"""

# Create annotations directory
mkdir -p /home/Aaron/SlowFast/ava_annotations

cd /home/Aaron/SlowFast/ava_annotations

echo "Downloading AVA annotation files..."

# Download label map
wget https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -O ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt

# Download validation annotations
wget https://research.google.com/ava/download/ava_val_v2.2.csv

# Download excluded timestamps
wget https://research.google.com/ava/download/ava_val_excluded_timestamps_v2.1.csv

# Download frame lists
mkdir -p frame_lists
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv -O frame_lists/val.csv

echo "AVA annotation files downloaded to /home/Aaron/SlowFast/ava_annotations/"
echo ""
echo "Files downloaded:"
echo "  - ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt"
echo "  - ava_val_v2.2.csv"
echo "  - ava_val_excluded_timestamps_v2.1.csv"
echo "  - frame_lists/val.csv"