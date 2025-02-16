import pytest
import torch
import os
from src.models.diffusion_model import TextToImageGenerator
from src.models.cnn_feature_extractor import ResNetFeatureExtractor
from src.models.svm_classifier import SVMClassifier

@pytest.fixture
def temp_img_dir(tmp_path):
    return tmp_path / "test_images"

def test_diffusion_model_generation(temp_img_dir):
    os.makedirs(temp_img_dir, exist_ok=True)
    generator = TextToImageGenerator()
    test_text = "测试生成"
    
    # 生成单张图片
    img_path = os.path.join(temp_img_dir, "test.png")
    generator.generate_image(test_text, img_path, steps=20)
    
    assert os.path.exists(img_path), "图片生成失败"
    assert os.path.getsize(img_path) > 1024, "生成的图片文件过小"

def test_cnn_feature_shape():
    model = ResNetFeatureExtractor()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (1, 512), f"预期特征维度是(1, 512)，实际是{output.shape}"

def test_svm_training():
    import numpy as np
    X = np.random.randn(100, 512)
    y = np.random.randint(0, 3, 100)
    
    svm = SVMClassifier()
    svm.train(X, y)
    
    assert hasattr(svm, 'grid_search'), "网格搜索未执行"
    assert svm.grid_search.best_params_ is not None, "最佳参数未找到"
    assert svm.grid_search.score(X, y) > 0.5, "模型预测准确率异常"
