import yaml

params = {
    "ALL_ATTR": [                               # 所有可选属性名称
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
        "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
        "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
        "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
        "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
        "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    ],
    "processed_file": "dataset",  # Autoencoder的优化器配置
    "batch_size": 32,                           # 每次训练的样本数
    "learning_rate": 1e-3,                           # 每次训练的样本数
    "total_epochs": 10000,                       # 总训练轮数
    "target_attribute_list": "ALL",
    "n_attributes": 40,


    # "autoencoder_optimizer": "adam,lr=0.0002",  # Autoencoder的优化器配置
    # "autoencoder_reload_path": None,            # 预训练Autoencoder模型的加载路径
    # "attribute_list": [["Smiling", 2]],         # 需要操作的属性列表及类别数量
    # "batch_size": 32,                           # 每次训练的样本数
    # "gradient_clip_norm": 5,                    # 梯度裁剪的最大范数
    # "debug_mode": False,                        # 是否启用调试模式
    # "decoder_dropout_rate": 0.0,                # 解码器中的Dropout概率
    # "decoder_upsample_method": "convtranspose", # 解码器的上采样方式
    # "discriminator_optimizer": "adam,lr=0.0002",# 判别器的优化器配置
    # "model_output_path": "/home/mla/Fader/models/default/0gil7kitln", # 模型输出路径
    # "samples_per_epoch": 50000,                 # 每个epoch的样本数量
    # "evaluation_classifier_path": "models/default/fxkbsjtp77/best.pth", # 外部分类器路径
    # "horizontal_flip_enabled": True,            # 是否启用水平翻转数据增强
    # "hidden_layer_dim": 512,                    # 隐藏层的维度大小
    # "image_channels": 3,                        # 图像的通道数（1: 灰度, 3: RGB）
    "image_size": 256,                          # 图像的宽度和高度（正方形）
    # "encoder_initial_filters": 32,              # 编码器的初始卷积核数量
    # "use_instance_normalization": False,        # 是否使用Instance Normalization
    # "autoencoder_loss_weight": 1,               # 自编码器损失的权重
    # "latent_discriminator_loss_weight": 0.0001, # 潜空间判别器损失权重
    # "discriminator_lambda_schedule_steps": 500000, # 判别器lambda调度的总步数
    # "latent_discriminator_dropout_rate": 0.3,   # 潜空间判别器的Dropout概率
    # "latent_discriminator_reload_path": None,   # 预训练潜空间判别器的加载路径
    # "maximum_feature_maps": 512,                # 卷积层的最大特征图数量
    # "attribute_count": 2,                       # 属性的总数量
    # "total_epochs": 1000,                       # 总训练轮数
    # "latent_discriminator_steps": 1,            # 每轮潜空间判别器的训练步数
    # "encoder_decoder_layers": 7,                # 编码器/解码器的层数
    # "experiment_name": "default",               # 实验名称
    # "ALL_ATTR": [                               # 所有可选属性名称
    #     "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    #     "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    #     "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    #     "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    #     "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    #     "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    #     "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    #     "Wearing_Necklace", "Wearing_Necktie", "Young"
    # ]
}

# 保存为 YAML 文件
with open("parameters_classifier.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False)