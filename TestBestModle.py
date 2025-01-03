import torch
from torchvision import transforms
import os
from PIL import Image
from dataset.dataset_preprocess import preprocess_and_save_dataset
from src.tools import CelebADataset, Config
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import yaml

# ========================
# 配置和初始化
# ========================
# 加载配置文件
config_file = "parameter/parameters_test.yaml"
with open(config_file, "r") as f:
    params_dict = yaml.safe_load(f)

params = Config(params_dict)

# 模型路径和测试数据路径
model_path = "best_model.pth"
test_data_path = params.preprocess_save_directory + "/test_dataset.pth"

# 属性名称（从配置中获取或指定）
attribute_names = params.target_attribute_list  # 目标属性名称列表

# 加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 加载分类器模型
# ========================
# 加载模型
model = torch.load(model_path, map_location=device)
model.eval()  # 设置为评估模式

# ========================
# 加载测试数据
# ========================
# 从 test_dataset.pth 文件加载数据
test_data = torch.load(test_data_path)  # 包含图像和标签
images, labels = test_data["images"], test_data["labels"]

# 随机选取 20 张图片及其对应的标签
indices = random.sample(range(len(images)), 20)
sampled_images = images[indices]
sampled_labels = labels[indices]

# ========================
# 模型预测
# ========================
# 将图片和标签移至设备
sampled_images = sampled_images.to(device)
sampled_labels = sampled_labels.to(device)

# 模型预测
outputs = model(sampled_images)  # 模型输出 (batch_size, n_attributes, num_classes)
predictions = torch.argmax(outputs, dim=-1)  # 获取每个属性的预测类别
true_labels = torch.argmax(sampled_labels, dim=-1)  # 将独热编码标签转换为整数类别

# ========================
# 准确率计算
# ========================
# 计算每个属性的准确率
attribute_accuracies = []
for attr_idx, attr_name in enumerate(attribute_names):
    acc = accuracy_score(true_labels[:, attr_idx].cpu(), predictions[:, attr_idx].cpu())
    attribute_accuracies.append((attr_name, acc))
    print(f"Accuracy for {attr_name}: {acc:.4f}")

# 平均准确率
mean_accuracy = sum([acc for _, acc in attribute_accuracies]) / len(attribute_accuracies)
print(f"Mean Accuracy: {mean_accuracy:.4f}")

# ========================
# 可视化部分结果
# ========================
# 可视化选取的 20 张图片及其预测
for i in range(20):
    image = sampled_images[i].permute(1, 2, 0).cpu().numpy()  # 转为 HWC 格式
    predicted_attrs = [attribute_names[j] for j in range(len(attribute_names)) if predictions[i, j] == 1]
    true_attrs = [attribute_names[j] for j in range(len(attribute_names)) if true_labels[i, j] == 1]

    # 显示图片及预测结果
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {', '.join(predicted_attrs)}\nTrue: {', '.join(true_attrs)}")
    plt.show()
