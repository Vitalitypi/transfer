from PIL import Image
from torchvision import transforms
import torch
from model import UNet
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
# 加载测试图像
test_image = Image.open('dataset/infer_input/0.jpg')
test_image = transform(test_image).unsqueeze(0).to(device)
model = UNet()
model = model.to(device)
state = torch.load('./trained/model.pth')
model.load_state_dict(state)
# 推理
model.eval()
with torch.no_grad():
    output = model(test_image)

# 转换为可视化格式
output_image = output.squeeze(0).cpu().permute(1, 2, 0).numpy()

# 展示结果
plt.imshow(output_image)
plt.show()
