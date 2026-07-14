import gc
import torch

from diffusers import (
    DiffusionPipeline,
    StableVideoDiffusionPipeline
)

from transformers import pipeline
from kokoro import KPipeline


class ModelManager:

    def __init__(self):

        self.current_model = None
        self.current_name = None

    # -------------------------------------------------------
    # Memory Cleanup
    # -------------------------------------------------------

    def unload_current_model(self):

        if self.current_model is not None:

            print(f"Unloading {self.current_name}...")

            try:
                if hasattr(self.current_model, "to"):
                    self.current_model.to("cpu")
            except Exception:
                pass

            del self.current_model

            self.current_model = None
            self.current_name = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------
    # Text to Image
    # -------------------------------------------------------

    def get_text_to_image(self):

        if self.current_name == "text_to_image":
            return self.current_model

        self.unload_current_model()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading Stable Diffusion...")

        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        pipe.to(device)

        if device == "cuda":
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()

            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()

        self.current_model = pipe
        self.current_name = "text_to_image"

        return pipe

    # -------------------------------------------------------
    # Image to Video
    # -------------------------------------------------------

    def get_image_to_video(self):

        if self.current_name == "image_to_video":
            return self.current_model

        self.unload_current_model()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading Stable Video Diffusion...")

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        pipe.to(device)

        if device == "cuda":
            pipe.enable_attention_slicing()

        self.current_model = pipe
        self.current_name = "image_to_video"

        return pipe

    # -------------------------------------------------------
    # Depth Estimation
    # -------------------------------------------------------

    def get_depth_estimation(self):

        if self.current_name == "depth_estimation":
            return self.current_model

        self.unload_current_model()

        device = 0 if torch.cuda.is_available() else -1

        print("Loading Depth Estimation Model...")

        depth_pipe = pipeline(
            task="depth-estimation",
            model="Intel/dpt-large",
            device=device
        )

        self.current_model = depth_pipe
        self.current_name = "depth_estimation"

        return depth_pipe

    # -------------------------------------------------------
    # Text to Speech
    # -------------------------------------------------------

    def get_text_to_speech(self):

        if self.current_name == "text_to_speech":
            return self.current_model

        self.unload_current_model()

        print("Loading Kokoro...")

        tts = KPipeline(lang_code="a")

        self.current_model = tts
        self.current_name = "text_to_speech"

        return tts


# -------------------------------------------------------
# Singleton Instance
# -------------------------------------------------------

model_manager = ModelManager()