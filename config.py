"""
 @Description: config.py in Handwritten-Digit-Recognition
 @Author: Jerry Huang
 @Date: 3/24/22 8:38 PM
"""
import torch

BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2