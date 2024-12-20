import yaml

# 参数配置字典
params = {
    "image_size": 256,  # 输入图像的大小 (宽和高，图像是正方形)
    "processed_file": "dataset",  # 处理后的数据文件
    "n_attributes": 1, # 训练的的属性数量
    "learning_rate": 1e-3,
    "ALL_ATTR": [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
        "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
        "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
        "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
        "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
        "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    ]                               # 所有可选属性名称
}

# 保存为 YAML 文件
with open("parameters.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False)


