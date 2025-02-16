import torch
from diffusers import StableDiffusionPipeline

class TextToImageGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-2-base"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        
    def generate_image(self, text, save_path, steps=50):
        image = self.pipe(
            text, 
            num_inference_steps=steps
        ).images[0]
        image.save(save_path)