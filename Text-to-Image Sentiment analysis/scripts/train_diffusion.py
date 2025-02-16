import yaml
from src.models.diffusion_model import TextToImageGenerator
from src.utils.data_loader import TextDataset

def main():
    # Load config
    with open("../configs/diffusion_config.yml") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    dataset = TextDataset(config['data_path'])
    generator = TextToImageGenerator()
    
    # Generate images
    for idx, text in enumerate(dataset):
        save_path = f"../results/generated_images/{idx}.png"
        generator.generate_image(text, save_path, steps=config['steps'])