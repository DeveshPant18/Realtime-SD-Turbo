import torch
import gradio as gr
# Import the custom function from our new pipeline.py file
from pipeline import load_sd_turbo_pipeline

# --- 1. Setup and Model Loading ---
# All the complex loading logic is now handled by our imported function.
print("🚀 Starting up...")
pipe = load_sd_turbo_pipeline()


# --- 2. The Image Generation Function ---
# This function remains the same as before.
def generate_image(prompt):
    """
    Generates an image from a text prompt using the loaded pipeline.
    """
    # The magic of SD-Turbo: 1 step is enough for a good image.
    image = pipe(
        prompt=prompt, 
        num_inference_steps=1, 
        guidance_scale=0.0
    ).images[0]
    
    return image


# --- 3. The Gradio User Interface ---
# This UI code also remains the same.
with gr.Blocks(title="⚡ Real-Time Image Generation") as iface:
    gr.Markdown("# ⚡ Real-Time Image Generation with SD-Turbo")
    gr.Markdown("Type a prompt and watch the image update instantly. This pipeline was built manually!")
    
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Your Prompt",
            placeholder="e.g., A cinematic photo of a raccoon wearing a space suit"
        )
    
    with gr.Row():
        image_output = gr.Image(label="Generated Image")

    # The real-time update mechanism also remains unchanged.
    prompt_input.input(
        fn=generate_image,
        inputs=prompt_input,
        outputs=image_output
    )

# --- 4. Launch the Application ---
if __name__ == "__main__":
    iface.launch()