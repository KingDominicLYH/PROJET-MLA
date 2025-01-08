import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# 根据您的项目结构，引入 Config（如有）和 AutoEncoder
from src.tools import Config  # 确保存在该文件与类
from src.models import AutoEncoder

# ======== 在这里设置您关注的属性、原图列表等 ========

raw_imgs = ["001007.jpg"]  # 这里可以放更多的图名，例如 ["001007.jpg", "000008.jpg", ...]

# 从 YAML 里读取训练时的参数，保证一致
with open("parameter/parameters.yaml", "r") as f:
    params_dict = yaml.safe_load(f)

params = Config(params_dict)  # 将YAML配置字典转换为Config对象
params.target_attribute_list = ["Smiling"]

# 1. 加载模型函数
def load_trained_autoencoder(model_path: str, device: torch.device, params):
    """
    加载已经训练好的AutoEncoder模型。
    """
    # 从文件加载权重
    checkpoint = torch.load(model_path, map_location=device)

    # 初始化模型并加载权重
    model = AutoEncoder(params).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model

# 2. 单张图像预处理
def load_and_preprocess_image(img_path: str, img_size: int = 256):
    """
    读取并预处理单张图像:
      - BGR->RGB
      - 中心裁剪(若训练时做过)
      - 缩放至(img_size, img_size)
      - 转换到 [-1, 1] 张量
    """
    # 读取图像(BGR)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 若您训练时裁掉了上下各 20 行，则请保持一致
    # 假设您原先写的就是 image[20:-20, :, :]
    if image.shape[0] > 40:
        image = image[20:-20, :, :]

    # 缩放至目标大小
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

    # 转换为张量，且归一化到[-1,1]
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()  # [3, H, W]
    image = image / 255.0  # [0,1]
    image = image * 2.0 - 1.0  # [-1,1]

    # 增加batch维度 [1, 3, H, W]
    image = image.unsqueeze(0)
    return image

# 3. 插值核心函数
def generate_attribute_interpolations(
    model: AutoEncoder,
    input_image: torch.Tensor,
    device: torch.device,
    n_interpolations: int = 10
):
    """
    对单张图像做属性插值，并返回 (n_interpolations) 张合成图像。
    假设只有一个属性(n_attributes=1)，则属性向量形状为 [1, 2]。
    例如从 [1,0] 到 [0,1] 线性插值。
    """
    input_image = input_image.to(device)

    # 提取图像 latent
    with torch.no_grad():
        latent = model.encoder(input_image)  # [1, 512, h', w']

    # alpha 从 0 到 1
    alphas = np.linspace(0, 1, n_interpolations)

    output_images = []
    with torch.no_grad():
        for alpha in alphas:
            # 单属性插值： attribute = [1 - alpha, alpha]
            attribute = torch.tensor([[1.0 - alpha, alpha]], device=device, dtype=torch.float32)  # 确保 attribute 为 float32
            # 解码
            out = model.decoder(latent, attribute)
            output_images.append(out.cpu())

    output_images = torch.cat(output_images, dim=0)  # [n_interpolations, 3, H, W]
    return output_images

# 4. 主逻辑：对列表中的每张图做插值
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载已训练好的模型（请根据实际路径修改）
    model_path = "train_model/best_autoencoder_Smiling.pth"
    autoencoder = load_trained_autoencoder(model_path, device, params)

    # 逐张图像进行插值
    n_interpolations = 10  # 生成多少个插值点
    for img_name in raw_imgs:
        # 拼接图像路径，如果在同一目录就直接用 img_name
        # 如果需要子目录请自行替换
        test_img_path = r"dataset\img_align_celeba\001007.jpg"
        # test_img_path = os.path.join(params.raw_img_directory, img_name)

        # 检查路径是否正确
        if not os.path.isfile(test_img_path):
            print(f"Image file not found: {test_img_path}")
            continue

        # 预处理
        input_image = load_and_preprocess_image(test_img_path, img_size=256)

        # 插值
        output_images = generate_attribute_interpolations(
            model=autoencoder,
            input_image=input_image,
            device=device,
            n_interpolations=n_interpolations
        )

        # 拼成一张网格
        # 让每张图占一列 => nrow=n_interpolations
        # （5）可视化：拼成一张网格图并保存
        # 手动归一化 output_images 到 [0, 1]
        output_images = (output_images - output_images.min()) / (output_images.max() - output_images.min())

        # 创建图像网格，不使用 range 参数
        grid = make_grid(output_images, nrow=n_interpolations)

        # 保存结果
        save_filename = f"interpolation_{os.path.splitext(img_name)[0]}.png"
        save_image(grid, save_filename)
        print(f"Interpolation result of {img_name} saved to: {save_filename}")

        # 如果想直接在matplotlib里可视化
        grid_np = grid.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
        plt.figure(figsize=(20, 4))  # 宽大一些
        plt.imshow(grid_np)
        plt.title(f"Interpolations of {img_name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
