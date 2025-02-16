import torch
import argparse
import yaml
from src.models.diffusion_model import TextToImageGenerator
from src.models.cnn_feature_extractor import ResNetFeatureExtractor
from src.models.svm_classifier import SVMClassifier
from src.utils.visualization import plot_generated_samples
import joblib

def main(args):
    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 初始化组件
    generator = TextToImageGenerator()
    cnn_model = ResNetFeatureExtractor()
    svm = joblib.load(config['svm_model_path'])
    
    # 输入文本预处理
    input_text = args.text if args.text else input("请输入评论内容: ")
    
    # 生成图像
    img_path = "temp_generated_image.png"
    generator.generate_image(input_text, img_path)
    
    # 提取特征
    transform = torch.nn.Sequential(
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    )
    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        features = cnn_model(image)
    
    # 分类预测
    pred_prob = svm.predict_proba(features.numpy())
    pred_class = svm.predict(features.numpy())[0]
    
    # 显示结果
    print(f"\n预测结果: {config['class_names'][pred_class]}")
    print("各类别概率:")
    for cls, prob in zip(config['class_names'], pred_prob[0]):
        print(f"- {cls}: {prob*100:.1f}%")
    
    # 可视化生成的图像
    plot_generated_samples([img_path], [input_text])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
        default='configs/inference_config.yml',
        help='配置文件路径'
    )
    parser.add_argument('--text', 
        type=str, 
        help='直接输入评论文本（可选）'
    )
    args = parser.parse_args()
    main(args)
