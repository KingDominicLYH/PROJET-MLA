import yaml

# 参数配置字典
params = {
    "image_size": 256,  # 输入图像的大小 (宽和高，图像是正方形)
    "processed_file": "dataset",  # 处理后的数据文件
    "n_attributes": 1,  # 训练的的属性数量
    "learning_rate": 1e-3,  # 学习率
    "raw_img_directory": "/home/mla/PROJET-MLA/dataset/img_align_celeba",  # 图像的原始目录路径
    "raw_attributes_file": "/home/mla/PROJET-MLA/dataset/list_attr_celeba.txt",  # 原始属性文件路径
    "preprocess_save_directory": "/home/mla/PROJET-MLA/dataset",  # 预处理保存目录
    "target_attribute_list": ["Male"],  # 目标属性列表
    "ALL_ATTR": [  # 所有可选属性名称
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
        "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
        "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
        "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
        "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
        "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    ],
    "model_path": "classifier_model/best_model.pth",
    "model_output_path": "model",
    "total_train_samples": 50000,
    "total_epochs": 1000,
    "batch_size": 32  # 批量大小
}

# 保存为 YAML 文件
with open("parameters.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False)


