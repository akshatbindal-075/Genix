import os
import gc
import uuid
import torch
import cv2
import numpy as np
from .model_manager import model_manager
 
from PIL import Image
from transformers import pipeline
 
# -------------------------------------------------------
# CUDA Memory Configuration
# -------------------------------------------------------
 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
 
# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------
 
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
 
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# -------------------------------------------------------
# Depth Estimation
# -------------------------------------------------------
 
def predict_depth(image_path):
 
    gc.collect()
 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    device = 0 if torch.cuda.is_available() else -1
 
    print(f"Running on: {'CUDA' if device == 0 else 'CPU'}")
 
    pipe = model_manager.get_depth_estimation()
 
    try:
 
        image = Image.open(image_path).convert("RGB")
 
        with torch.inference_mode():
 
            result = pipe(image)
 
        depth = np.array(result["depth"])
 
        depth = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )
 
        depth = depth.astype(np.uint8)
 
        colored = cv2.applyColorMap(
            depth,
            cv2.COLORMAP_INFERNO
        )
 
        filename = f"depth_map_{uuid.uuid4().hex}.png"
 
        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )
 
        cv2.imwrite(output_path, colored)
 
        return output_path
 
    finally:
 
        print("Releasing GPU Memory...")
 
        del pipe
 
        gc.collect()
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
 
# -------------------------------------------------------
# Local Testing
# -------------------------------------------------------
 
if __name__ == "__main__":
 
    image_path = input("Enter Image Path:\n")
 
    output = predict_depth(image_path)
 
    print("\nDepth Map Generated Successfully!")
    print(output)