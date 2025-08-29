import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

def load_sd_turbo_pipeline():
    """
    Manually loads all components and assembles the SD-Turbo pipeline.
    This provides more control and clarity on the model's architecture.
    """
    # --- Configuration ---
    model_id = "stabilityai/sd-turbo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print("🔧 Building pipeline from individual components...")
    print(f"Using device: {device}")

    # --- Load Individual Components from Hugging Face ---
    
    # 1. VAE (Autoencoder): Decodes the image from latent space.
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype)

    # 2. Text Encoder & Tokenizer: Understands the text prompt.
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    # 3. U-Net: The core model that denoises the image.
   
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch_dtype)

    # 4. Scheduler: Manages the denoising steps.
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # --- Assemble the Pipeline ---
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    
    # Move the entire pipeline to the selected device (GPU or CPU).
    pipe.to(device)
    
    print("✅ Pipeline built successfully!")
    return pipe

if __name__ == '__main__':
    # This part is for testing the pipeline directly if you run "python pipeline.py"
    print("Running a direct test of the pipeline build process...")
    pipeline = load_sd_turbo_pipeline()
    print("Test complete. Pipeline object created:", pipeline)

