"""
Genix - Multi-Model Generative AI Playground

This application integrates four core AI modules:
1. Text-to-Image (Stable Diffusion XL)
2. Text-to-Speech (Kokoro-82M)
3. Image-to-Video (Stable Video Diffusion img2vid-xt)
4. Depth Estimation (Intel DPT-Large)

Built with Gradio Blocks and customized with a modern dark theme.
"""

import os
import gradio as gr

# Import AI modules
from modules.text_to_image import generate_image
from modules.text_to_speech import generate_audio
from modules.image_to_video import generate_video
from modules.depth_estimation import predict_depth


# -------------------------------------------------------
# Custom CSS Styling
# -------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    --primary-glow: 0 4px 20px rgba(139, 92, 246, 0.35);
    --border-color: #334155;
    --hover-glow: 0 8px 30px rgba(139, 92, 246, 0.5);
}

body, .gradio-container {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header UI */
.header-container {
    text-align: center;
    padding: 3rem 1.5rem;
    margin-bottom: 2.5rem;
    background: radial-gradient(circle at top, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.glow-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
    letter-spacing: -0.04em;
    filter: drop-shadow(0 2px 8px rgba(139, 92, 246, 0.3));
}

.subtitle {
    font-size: 1.25rem;
    color: #94a3b8;
    font-weight: 400;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Tabs UI Navigation */
.custom-tabs .tab-nav {
    border-bottom: 1px solid var(--border-color) !important;
    gap: 0.5rem !important;
}

.custom-tabs .tab-nav button {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 1.5rem !important;
    border-radius: 12px 12px 0 0 !important;
    color: #94a3b8 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: transparent !important;
    border: none !important;
}

.custom-tabs .tab-nav button.selected {
    color: #c084fc !important;
    border-bottom: 3px solid #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.08) !important;
}

.custom-tabs .tab-nav button:hover:not(.selected) {
    color: #f1f5f9 !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

/* Action Button styling */
.action-btn {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 14px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: var(--primary-glow) !important;
    cursor: pointer !important;
}

.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--hover-glow) !important;
    filter: brightness(1.1) !important;
}

.action-btn:active {
    transform: translateY(0) !important;
    box-shadow: var(--primary-glow) !important;
}

/* Status Notifications container styling */
.status-msg {
    margin-top: 1rem !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
}

.status-success {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    color: #34d399 !important;
}

/* Footer UI */
.footer-container {
    text-align: center;
    margin-top: 4rem;
    padding: 2rem 1.5rem;
    border-top: 1px solid var(--border-color);
    color: #64748b;
    font-size: 0.95rem;
}

.footer-tags {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}

.footer-tag {
    background: #1e293b;
    color: #cbd5e1;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid var(--border-color);
    transition: all 0.2s ease;
}

.footer-tag:hover {
    background: #334155;
    color: #f1f5f9;
}
"""

# JS script to dynamically add dark mode class on page mount
FORCE_DARK_MODE_JS = """
() => {
    document.documentElement.classList.add('dark');
}
"""

# Define Gradio Theme (Primary Violet, Neutral Slate)
theme = gr.themes.Default(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_800",
    block_border_color_dark="*neutral_800",
)

# -------------------------------------------------------
# Inference Wrapper Functions
# -------------------------------------------------------

def text_to_image_interface(prompt, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper for Text-to-Image generation module.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a valid image prompt.")
    try:
        gr.Info("Launching Image Generation...")
        output_path = generate_image(prompt)
        gr.Info("Image successfully generated!")
        return output_path, gr.update(
            value="✨ **Success**: Image generated successfully!",
            visible=True,
            elem_classes=["status-msg", "status-success"]
        )
    except Exception as e:
        raise gr.Error(f"Image generation failed: {str(e)}")


def text_to_speech_interface(text, voice, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper for Text-to-Speech generation module.
    """
    if not text or not text.strip():
        raise gr.Error("Please enter text to generate speech.")
    try:
        gr.Info("Launching Speech Generation...")
        output_path = generate_audio(text, voice=voice)
        gr.Info("Speech successfully generated!")
        return output_path, gr.update(
            value="✨ **Success**: Speech generated successfully!",
            visible=True,
            elem_classes=["status-msg", "status-success"]
        )
    except Exception as e:
        raise gr.Error(f"Speech generation failed: {str(e)}")


def image_to_video_interface(image, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper for Image-to-Video generation module.
    """
    if image is None:
        raise gr.Error("Please upload an input image.")
    try:
        gr.Info("Launching Video Generation (this may take a minute)...")
        output_path = generate_video(image)
        gr.Info("Video successfully generated!")
        return output_path, gr.update(
            value="✨ **Success**: Video generated successfully!",
            visible=True,
            elem_classes=["status-msg", "status-success"]
        )
    except Exception as e:
        raise gr.Error(f"Video generation failed: {str(e)}")


def depth_estimation_interface(image, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper for Depth Estimation module.
    """
    if image is None:
        raise gr.Error("Please upload an input image.")
    try:
        gr.Info("Launching Depth Estimation...")
        output_path = predict_depth(image)
        gr.Info("Depth estimation completed!")
        return output_path, gr.update(
            value="✨ **Success**: Depth estimation completed successfully!",
            visible=True,
            elem_classes=["status-msg", "status-success"]
        )
    except Exception as e:
        raise gr.Error(f"Depth estimation failed: {str(e)}")

# -------------------------------------------------------
# Gradio Blocks Layout Design
# -------------------------------------------------------

with gr.Blocks(css=CUSTOM_CSS, theme=theme, title="Genix") as demo:
    # Header Section
    gr.HTML("""
        <div class="header-container">
            <h1 class="glow-title">Genix</h1>
            <p class="subtitle">A Multi-Model Generative AI Playground powered by Hugging Face.</p>
        </div>
    """)

    # Main Tabs Container
    with gr.Tabs(elem_classes="custom-tabs"):
        
        # Tab 1: Text to Image
        with gr.TabItem("Text to Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Image Prompt",
                        placeholder="Enter your image prompt...",
                        lines=3
                    )
                    gen_img_btn = gr.Button("Generate Image", elem_classes=["action-btn"])
                    
                    # Example prompt
                    gr.Examples(
                        examples=[["A futuristic cyberpunk city at night with neon lights and flying cars, digital art"]],
                        inputs=prompt_input
                    )
                    
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Generated Image", type="filepath")
                    img_status = gr.Markdown(value="", visible=False)
            
            # Events
            gen_img_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=img_status
            ).then(
                fn=text_to_image_interface,
                inputs=prompt_input,
                outputs=[img_output, img_status]
            )

        # Tab 2: Text to Speech
        with gr.TabItem("Text to Speech"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Text Input",
                        placeholder="Enter text...",
                        lines=3
                    )
                    voice_input = gr.Dropdown(
                        label="Voice Character",
                        choices=["af_bella", "af_sarah", "am_adam", "bf_emma", "bm_george"],
                        value="af_bella"
                    )
                    gen_audio_btn = gr.Button("Generate Speech", elem_classes=["action-btn"])
                    
                    # Example text
                    gr.Examples(
                        examples=[["Welcome to Genix. This is a multi-model generative playground powered by Gradio."]],
                        inputs=text_input
                    )
                    
                with gr.Column(scale=1):
                    audio_output = gr.Audio(label="Generated Speech", type="filepath")
                    audio_status = gr.Markdown(value="", visible=False)
            
            # Events
            gen_audio_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=audio_status
            ).then(
                fn=text_to_speech_interface,
                inputs=[text_input, voice_input],
                outputs=[audio_output, audio_status]
            )

        # Tab 3: Image to Video
        with gr.TabItem("Image to Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_to_vid_input = gr.Image(label="Source Image", type="filepath")
                    gen_vid_btn = gr.Button("Generate Video", elem_classes=["action-btn"])
                    
                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Generated Video")
                    vid_status = gr.Markdown(value="", visible=False)
            
            # Events
            gen_vid_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=vid_status
            ).then(
                fn=image_to_video_interface,
                inputs=img_to_vid_input,
                outputs=[vid_output, vid_status]
            )

        # Tab 4: Depth Estimation
        with gr.TabItem("Depth Estimation"):
            with gr.Row():
                with gr.Column(scale=1):
                    depth_img_input = gr.Image(label="Source Image", type="filepath")
                    est_depth_btn = gr.Button("Estimate Depth", elem_classes=["action-btn"])
                    
                with gr.Column(scale=1):
                    depth_output = gr.Image(label="Estimated Depth Map", type="filepath")
                    depth_status = gr.Markdown(value="", visible=False)
            
            # Events
            est_depth_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=depth_status
            ).then(
                fn=depth_estimation_interface,
                inputs=depth_img_input,
                outputs=[depth_output, depth_status]
            )

    # Footer Section
    gr.HTML("""
        <div class="footer-container">
            <p>Built with ❤️ using the best generative AI tech stack:</p>
            <div class="footer-tags">
                <span class="footer-tag">Python</span>
                <span class="footer-tag">Gradio</span>
                <span class="footer-tag">Hugging Face</span>
                <span class="footer-tag">Diffusers</span>
                <span class="footer-tag">Transformers</span>
            </div>
        </div>
    """)

    # Apply dark mode immediately on load
    demo.load(None, js=FORCE_DARK_MODE_JS)

# -------------------------------------------------------
# Run Locally
# -------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True
    )
