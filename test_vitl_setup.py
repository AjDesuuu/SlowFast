#!/usr/bin/env python3
"""
Test ViT-L model loading and basic functionality before running full evaluation
"""

import sys
import os
import torch
sys.path.append('/home/Aaron/SlowFast')

def test_checkpoint_loading():
    """Test loading the ViT-L checkpoint"""
    print("Testing ViT-L checkpoint loading...")
    
    checkpoint_path = "/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"✅ Checkpoint found: {checkpoint_path}")
    print(f"📊 Checkpoint size: {os.path.getsize(checkpoint_path) / (1024**3):.2f} GB")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        # Check checkpoint structure
        print("📋 Checkpoint keys:", list(checkpoint.keys()))
        
        if 'target_encoder' in checkpoint:
            encoder_state = checkpoint['target_encoder']
            print(f"📊 Target encoder parameters: {len(encoder_state)} keys")
            
            # Check some key parameters
            sample_keys = list(encoder_state.keys())[:5]
            print(f"🔑 Sample parameter keys: {sample_keys}")
            
            # Check if it looks like ViT-L
            for key in encoder_state.keys():
                if 'blocks' in key and 'weight' in key:
                    shape = encoder_state[key].shape
                    if len(shape) == 2 and shape[0] == 1024:  # ViT-L feature dim
                        print(f"✅ Found ViT-L signature: {key} with shape {shape}")
                        break
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def test_model_creation():
    """Test creating the ViT-L model"""
    print("\nTesting ViT-L model creation...")
    
    try:
        # Import our model
        from slowfast.models.vitl_model import ViTLAVAModel
        from slowfast.config.defaults import get_cfg
        
        # Create config
        cfg = get_cfg()
        cfg.MODEL.NUM_CLASSES = 80
        cfg.MODEL.DROPOUT_RATE = 0.5
        cfg.MODEL.HEAD_ACT = 'relu'
        cfg.DATA.TEST_CROP_SIZE = 224
        cfg.DATA.NUM_FRAMES = 4
        cfg.DETECTION.ENABLE = True
        cfg.DETECTION.ALIGNED = True
        cfg.DETECTION.SPATIAL_SCALE_FACTOR = 16
        cfg.DETECTION.ROI_XFORM_RESOLUTION = 7
        cfg.VITL_CHECKPOINT_PATH = "/home/Aaron/datasets/ava-dataset-utils/checkpoints/vitl16.pth.tar"
        
        # Create model
        model = ViTLAVAModel(cfg)
        print("✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🔓 Trainable parameters: {trainable_params:,}")
        print(f"🔒 Frozen parameters: {total_params - trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_paths():
    """Test if AVA data paths exist"""
    print("\nTesting AVA data paths...")
    
    paths_to_check = [
        "/home/Aaron/datasets/ava-dataset-utils/ava/frames",
        "/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists",
        "/home/Aaron/datasets/ava-dataset-utils/ava/annotations",
        "/home/Aaron/datasets/ava-dataset-utils/ava/frame_lists/val.csv",
        "/home/Aaron/datasets/ava-dataset-utils/ava/annotations/ava_val_v2.2.csv",
        "/home/Aaron/datasets/ava-dataset-utils/ava/annotations/ava_val_predicted_boxes.csv"
    ]
    
    all_good = True
    for path in paths_to_check:
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"✅ {path} (📁 {count} items)")
            else:
                size = os.path.getsize(path)
                print(f"✅ {path} (📄 {size:,} bytes)")
        else:
            print(f"❌ {path} - NOT FOUND")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🧪 Testing ViT-L AVA Evaluation Setup")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Checkpoint loading
    if test_checkpoint_loading():
        tests_passed += 1
    
    # Test 2: Model creation
    if test_model_creation():
        tests_passed += 1
    
    # Test 3: Data paths
    if test_data_paths():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready to run evaluation.")
        print("\n🚀 Run evaluation with:")
        print("cd /home/Aaron/SlowFast")
        print("python run_vitl_eval.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()