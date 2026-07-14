import os
import gc
import uuid
import torch
import imageio
from .model_manager import model_manager
 
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
 
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
# Generate Video
# -------------------------------------------------------
 
def generate_video(image_path):
 
    gc.collect()
 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    print(f"Running on: {device}")
 
    pipe = pipe = model_manager.get_image_to_video()
 
    pipe.to(device)
 
    if device == "cuda":
        pipe.enable_attention_slicing()
 
    try:
 
        image = Image.open(image_path).convert("RGB")
 
        # Lower resolution for RTX 3050 (6 GB)
        image = image.resize((512, 288))
 
        generator = torch.Generator(device=device).manual_seed(42)
 
        with torch.inference_mode():
 
            frames = pipe(
                image,
                num_frames=8,
                decode_chunk_size=1,
                generator=generator,
            ).frames[0]
 
        filename = f"image_to_video_{uuid.uuid4().hex}.mp4"
 
        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )
 
        imageio.mimsave(
            output_path,
            frames,
            fps=7
        )
 
        return output_path
 
    finally:
 
        print("Releasing GPU Memory...")
 
        pipe.to("cpu")
 
        del pipe
 
        gc.collect()
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
 
# -------------------------------------------------------
# Local Testing
# -------------------------------------------------------
 
if __name__ == "__main__":
 
    image_path = input("Enter Image Path:\n")
 
    video = generate_video(image_path)
 
    print("\nVideo Generated Successfully!")
    print(video)