from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = os.listdir(input_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.input_dir, self.images[idx]))
        target_image = Image.open(os.path.join(self.target_dir, self.images[idx]))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
def get_dataloader():
    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 创建数据集和数据加载器
    train_dataset = ImageDataset('D:\program\python\\transfer\dataset\input\\', 'D:\program\python\\transfer\dataset\\target\\', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return train_loader
