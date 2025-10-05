#!/usr/bin/env python3
"""
Test script to verify learning rate scheduler is working correctly
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Create a simple model and optimizer
model = nn.Linear(10, 1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Configuration values
WARMUP_EPOCHS = 2
MAX_EPOCH = 30

def lr_lambda(epoch):
    # epoch is 0-indexed, so we need to add 1 for proper warmup calculation
    current_epoch = epoch + 1
    if current_epoch <= WARMUP_EPOCHS:
        # Linear warmup from 0.1 to 1.0 of base LR
        warmup_factor = 0.1 + 0.9 * (current_epoch / WARMUP_EPOCHS)
        return warmup_factor
    else:
        # Cosine decay after warmup
        progress = (current_epoch - WARMUP_EPOCHS) / (MAX_EPOCH - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))

lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("Testing Learning Rate Scheduler:")
print("Base LR: 1e-4")
print("Warmup epochs: 2")
print("Max epochs: 30")
print()

for epoch in range(10):  # Test first 10 epochs
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    lr_scheduler.step()

print("\nThe learning rate should:")
print("- Start at 0.1 * base_lr = 0.00001 for epoch 0")  
print("- Reach 1.0 * base_lr = 0.0001 by epoch 1")
print("- Then decay with cosine schedule after epoch 2")