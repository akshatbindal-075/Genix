import os

import torch
from diffusers import (
    DiffusionPipeline,
    StableVideoDiffusionPipeline,
)
from transformers import pipeline
from kokoro import KPipeline

# -------------------------------------------------------
# Model Cache Directory
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("GENIX MODEL SETUP")
print("=" * 60)
print(f"\nModels will be stored in:\n{CACHE_DIR}")

# -------------------------------------------------------
# Text-to-Image
# -------------------------------------------------------

print("\n[1/4] Downloading Stable Diffusion v1.5...")

DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float32,
)

print("✓ Stable Diffusion downloaded.")

# -------------------------------------------------------
# Image-to-Video
# -------------------------------------------------------

print("\n[2/4] Downloading Stable Video Diffusion...")

StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float32,
)

print("✓ Stable Video Diffusion downloaded.")

# -------------------------------------------------------
# Depth Estimation
# -------------------------------------------------------

print("\n[3/4] Downloading Intel DPT Large...")

pipeline(
    task="depth-estimation",
    model="Intel/dpt-large",
    cache_dir=CACHE_DIR,
    device=-1,
)

print("✓ Intel DPT downloaded.")

# -------------------------------------------------------
# Text-to-Speech
# -------------------------------------------------------

print("\n[4/4] Downloading Kokoro...")

KPipeline(lang_code="a")

print("✓ Kokoro downloaded.")

# -------------------------------------------------------
# Finished
# -------------------------------------------------------

print("\n" + "=" * 60)
print("ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("=" * 60)

print("\nModels are stored in:")
print(CACHE_DIR)