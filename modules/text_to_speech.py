import os
import gc
import uuid
import torch
import soundfile as sf
 
from .model_manager import model_manager
from kokoro import KPipeline
 
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
# Text To Speech
# -------------------------------------------------------
 
def generate_audio(text, voice="af_bella"):
 
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
 
    gc.collect()
 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    print(f"Running on: {device}")
 
    pipeline = model_manager.get_text_to_speech()
 
    try:
 
        generator = pipeline(
            text,
            voice=voice
        )
 
        output_path = os.path.join(
            OUTPUT_DIR,
            f"speech_{uuid.uuid4().hex}.wav"
        )
 
        for _, (_, _, audio) in enumerate(generator):
 
            sf.write(
                output_path,
                audio,
                24000
            )
 
            return output_path
 
        raise RuntimeError("Audio generation failed.")
 
    finally:
 
        print("Releasing Memory...")
 
        del pipeline
 
        gc.collect()
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
 
# -------------------------------------------------------
# Local Testing
# -------------------------------------------------------
 
if __name__ == "__main__":
 
    text = input("Enter Text:\n")
 
    audio = generate_audio(text)
 
    print("\nAudio Generated Successfully!")
    print(audio)
