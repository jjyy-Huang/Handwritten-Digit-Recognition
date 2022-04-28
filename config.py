"""
 @Description: config.py in Handwritten-Digit-Recognition
 @Author: Jerry Huang
 @Date: 3/24/22 8:38 PM
"""
import torch

BATCH_SIZE = 128
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
INFER_MODE = 'float32'