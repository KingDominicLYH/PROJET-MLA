import yaml

params = {
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
    "save_dir": "classifier_model",  # Directory to save classifier model
    "processed_file": "dataset",  # Processed data file
    "batch_size": 32,  # Number of samples per batch
    "learning_rate": 1e-3,  # Learning rate
    "total_epochs": 1000,  # Total number of training epochs
    "target_attribute_list": "ALL",  # List of target attributes (set to "ALL")
    "n_attributes": 40,  # Total number of attributes
    "total_train_samples": 50000,  # Total number of training samples
    "total_valid_samples": 10000,  # Total number of validation samples
    "optimizer": "adam,lr=0.001",  # Optimizer configuration

    # Additional configurations (commented out)
    # "autoencoder_optimizer": "adam,lr=0.0002",  # Autoencoder optimizer
    # "autoencoder_reload_path": None,  # Pretrained autoencoder model path
    # "attribute_list": [["Smiling", 2]],  # Attributes to manipulate and their categories
    # "gradient_clip_norm": 5,  # Maximum norm for gradient clipping
    # "debug_mode": False,  # Enable debug mode
    # "decoder_dropout_rate": 0.3,  # Dropout probability in the decoder
    # "decoder_upsample_method": "convtranspose",  # Upsampling method in the decoder
    # "discriminator_optimizer": "adam,lr=0.0002",  # Discriminator optimizer configuration
    # "model_output_path": "/home/mla/Fader/models/default/0gil7kitln",  # Model output directory
    # "samples_per_epoch": 50000,  # Number of samples per epoch
    # "evaluation_classifier_path": "models/default/fxkbsjtp77/best.pth",  # External classifier path
    # "horizontal_flip_enabled": True,  # Enable horizontal flip data augmentation
    # "hidden_layer_dim": 512,  # Hidden layer dimension
    # "image_channels": 3,  # Number of image channels (1: grayscale, 3: RGB)
    "image_size": 256,  # Image width and height (square)
    # "encoder_initial_filters": 32,  # Initial number of convolution filters in the encoder
    # "use_instance_normalization": False,  # Use instance normalization
    # "autoencoder_loss_weight": 1,  # Weight for autoencoder loss
    # "latent_discriminator_loss_weight": 0.0001,  # Latent discriminator loss weight
    # "discriminator_lambda_schedule_steps": 500000,  # Steps for discriminator lambda scheduling
    # "latent_discriminator_dropout_rate": 0.3,  # Dropout probability in the latent discriminator
    # "latent_discriminator_reload_path": None,  # Pretrained latent discriminator model path
    # "maximum_feature_maps": 512,  # Maximum number of feature maps in convolution layers
    # "attribute_count": 2,  # Total number of attributes
    # "latent_discriminator_steps": 1,  # Number of training steps for the latent discriminator per epoch
    # "encoder_decoder_layers": 7,  # Number of encoder/decoder layers
    # "experiment_name": "default",  # Experiment name
}

# Save as a YAML file
with open("parameters_classifier.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False)
