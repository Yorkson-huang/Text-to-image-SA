import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.models import ResNetFeatureExtractor, SVMClassifier

def extract_features(dataloader, model):
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            features.append(outputs.numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

def main():
    # Feature extraction
    cnn_model = ResNetFeatureExtractor()
    train_loader = ... # 初始化数据加载器
    train_features, train_labels = extract_features(train_loader, cnn_model)
    
    # SVM Training
    svm = SVMClassifier()
    svm.train(train_features, train_labels)
    svm.save_model("../results/model_weights/best_svm.pkl")