#!/usr/bin/env python3
"""
Run ViT-L AVA evaluation using SlowFast's standard pipeline
"""

import sys
import os
sys.path.append('/home/Aaron/SlowFast')

def main():
    # Import after adding to path
    from tools.run_net import main as run_net_main
    import sys
    
    # Set up arguments for SlowFast's run_net.py
    original_argv = sys.argv[:]
    
    sys.argv = [
        'run_net.py',
        '--cfg', '/home/Aaron/SlowFast/configs/AVA/VITL_AVA.yaml',
        'TRAIN.ENABLE', 'False',
        'TEST.ENABLE', 'True',
        'NUM_GPUS', '1'
    ]
    
    try:
        run_net_main()
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()