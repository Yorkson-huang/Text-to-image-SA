import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class TextImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (str): 评论数据csv路径
            transform (callable): 图像预处理
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.image_dir = os.path.join(os.path.dirname(csv_path), 'images')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['review_text']
        label = row['sentiment']
        img_path = os.path.join(self.image_dir, f"{idx}.png")
        
        # 加载生成的图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return {
            'text': text,
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(csv_path, batch_size=32, split_ratio=0.8):
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = TextImageDataset(csv_path, transform=transform)
    
    # 分割训练集/测试集
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
