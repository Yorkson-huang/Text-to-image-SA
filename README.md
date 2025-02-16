# 🌟 Text-to-image-SA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange)](https://huggingface.co)

AI-powered E-commerce Sentiment Analysis with Text-to-Image Generation and CNN-SVM Ensemble

## ✨ Key Features

- **Text-to-Image Generation**: Visualize reviews using Stable Diffusion v2
- **Dual-Modal Analysis**: Combine textual and generated visual features
- **High-Efficiency Classifier**: Hybrid ResNet34 + SVM architecture
- **End-to-End Pipeline**: From raw text to final prediction
- **Config-Driven Workflow**: YAML configuration system

## 📂 Project Structure
```bash
├── configs/
│   ├── diffusion.yaml      # Text-to-image configuration
│   └── classifier.yaml     # Classification model config
├── src/
│   ├── models/             # Model definitions
│   ├── utils/              # Data processing tools
│   └── visualization/      # Visualization modules
├── data/
│   ├── raw/                # Raw review data
│   └── generated/          # Generated images
├── scripts/                # Execution scripts
├── docs/                   # Documentation
└── results/                # Output directory
```
## ⚡ Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-sentiment.git
cd multimodal-sentiment

# Install dependencies
pip install -r requirements.txt

# Authenticate Hugging Face (requires account)
huggingface-cli login
```
### Data Preparation
Sample CSV format (data/raw/reviews.csv):
```csv
text,label
"Product exceeds expectations, very satisfied!",2
"Poor packaging and slow shipping",0
...
Label mapping:
0 ➞ Negative
1 ➞ Neutral
2 ➞ Positive
```
### Basic Usage
1. Generate images from reviews:
```bash
python scripts/generate_images.py
python --config configs/diffusion.yaml
python --input data/raw/reviews.csv
python --output data/generated
   ```
2. Train classification model:
  ```bash
  python scripts/train_classifier.py 
  --config configs/classifier.yaml 
  --data_dir data/generated 
  --model_save results/models
  ```
3. Real-time inference:
 ```bash
 python scripts/inference.py 
 text "Poor customer service but decent product quality" 
 model_path results/models/best_model.pkl
 ```
## ⚙️ Configuration System
 ```yaml
model:
  name: "stabilityai/stable-diffusion-2"
  image_size: 512
  num_steps: 50
  
generation:
  batch_size: 4
  max_length: 77    # Max text token length
```
