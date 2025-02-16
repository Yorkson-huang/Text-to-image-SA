import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def plot_generated_samples(image_paths, titles, n_cols=4, save_path=None):
    """
    可视化生成的图像样本
    """
    n_rows = len(image_paths) // n_cols + 1
    plt.figure(figsize=(15, 5*n_rows))
    
    for i, (img_path, title) in enumerate(zip(image_paths, titles)):
        img = plt.imread(img_path)
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=8)
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def visualize_features(features, labels, save_path=None):
    """
    t-SNE特征可视化
    """
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1],
        hue=labels,
        palette="viridis",
        alpha=0.7
    )
    plt.title("Feature Space Visualization")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(true_labels, preds, classes, save_path=None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()
