import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# 定义自定义数据集
class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 数据集路径
input_dir = "./dataset/input"  # 替换为实际路径
target_dir = "./dataset/target"  # 替换为实际路径

# 加载数据集
dataset = ImageDataset(input_dir, target_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
optimizer = optim.Adam(generator.parameters(), lr=0.0002)
criterion = nn.MSELoss()

# 训练模型
epochs = 200
for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = generator(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 测试生成效果
generator.eval()
with torch.no_grad():
    sample_input, _ = dataset[0]  # 取第一个样本测试
    sample_input = sample_input.unsqueeze(0).to(device)
    generated_output = generator(sample_input).squeeze(0).cpu()
    generated_output = (generated_output * 0.5 + 0.5).clamp(0, 1)

    # 显示结果
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(transforms.ToPILImage()(sample_input.cpu().squeeze(0)))

    plt.subplot(1, 2, 2)
    plt.title("Generated Image")
    plt.imshow(transforms.ToPILImage()(generated_output))
    plt.show()
