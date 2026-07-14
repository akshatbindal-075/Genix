import os
import gc
import uuid
import torch
from diffusers import DiffusionPipeline
from .model_manager import model_manager
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
# Text-to-Image Function
# -------------------------------------------------------
 
def generate_image(prompt):
 
    gc.collect()
 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    print(f"Running on: {device}")
 
    pipe = model_manager.get_text_to_image()
    pipe.to(device)
 
    if device == "cuda":

        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
 
    try:
 
        with torch.inference_mode():
 
            image = pipe(
                prompt,
                height=512,
                width=512,
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]
 
        filename = f"text_to_image_{uuid.uuid4().hex}.png"
 
        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )
 
        image.save(output_path)
 
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
 
    prompt = input("Enter Prompt:\n")
 
    image_path = generate_image(prompt)
 
    print("\nImage Generated Successfully!")
    print(image_path)