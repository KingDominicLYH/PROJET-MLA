import yaml

# Parameter configuration dictionary
params = {
    "image_size": 256,  # Input image size (width and height, assuming square images)
    "processed_file": "dataset",  # Processed data file name
    "n_attributes": 1,  # Number of attributes for training
    "learning_rate": 1e-3,  # Learning rate
    "raw_img_directory": "/home/mla/PROJET-MLA/dataset/img_align_celeba",  # Path to the raw image directory
    "raw_attributes_file": "/home/mla/PROJET-MLA/dataset/list_attr_celeba.txt",  # Path to the raw attribute file
    "preprocess_save_directory": "/home/mla/PROJET-MLA/dataset",  # Directory to save preprocessed data
    "target_attribute_list": ["Male"],  # List of target attributes
    "ALL_ATTR": [  # List of all available attribute names
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
        "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
        "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
        "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
        "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
        "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    ],
    "model_path": "classifier_model/best_model.pth",  # Path to the trained model
    "model_output_path": "model",  # Directory to save model output
    "autoencoder_loss_weight": 1,  # Weight for autoencoder loss
    "discriminator_loss_weight": 0,  # Weight for discriminator loss
    "total_train_samples": 50000,  # Total number of training samples
    "total_epochs": 1000,  # Total number of training epochs
    "save_dir": "train_model",  # Directory to save training results
    "batch_size": 32  # Batch size for training
}

# Save parameters to a YAML file
with open("parameters.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False)



