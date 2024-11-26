import torch
from torch import nn

class Encoder(nn.Module):
    """
    Encoder module: Compresses an input image into a latent representation.
    """

    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 输入通道：3（RGB），输出通道：16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 2
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 3
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 5
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 6
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 7
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    """
    Decoder module: Reconstructs an image from a latent representation and attributes.
    """

    def __init__(self, n_attributes):
        super(Decoder, self).__init__()
        self.n_attributes = n_attributes
        attribute_channels = 2 * n_attributes  # Each attribute is represented as [1, 0] or [0, 1]

        # Define each group of ConvTranspose2d, BatchNorm2d, and Activation as a sequence
        self.layers = nn.ModuleList([
            # Layer 1
            nn.Sequential(
                nn.ConvTranspose2d(512 + attribute_channels, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),

            # Layer 2
            nn.Sequential(
                nn.ConvTranspose2d(512 + attribute_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),

            # Layer 3
            nn.Sequential(
                nn.ConvTranspose2d(256 + attribute_channels, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),

            # Layer 4
            nn.Sequential(
                nn.ConvTranspose2d(128 + attribute_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),

            # Layer 5
            nn.Sequential(
                nn.ConvTranspose2d(64 + attribute_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),

            # Layer 6
            nn.Sequential(
                nn.ConvTranspose2d(32 + attribute_channels, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),

            # Layer 7
            nn.Sequential(
                nn.ConvTranspose2d(16 + attribute_channels, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()  # Output normalized to the range [-1, 1]
            )
        ])

    def forward(self, z, attributes):
        """
        Forward pass through the decoder.

        Parameters:
        z: Latent representation from the encoder (shape: [batch_size, 512, 2, 2])
        attributes: Attribute vector (shape: [batch_size, n_attributes]), one-hot encoded
        """
        # Expand attribute vector to match spatial dimensions of the latent representation
        batch_size, _, h, w = z.shape

        attributes = attributes.reshape(batch_size, -1, 1, 1)  # Reshape to (batch_size, 2n, 1, 1)
        attributes = attributes.expand(batch_size, -1, h, w)  # Expand to (batch_size, 2n, h, w)

        # Initial input: concatenate latent representation and attributes
        x = torch.cat([z, attributes], dim=1)  # Shape: (batch_size, 512 + 2n, h, w)

        # Decode through each sequence
        for layer in self.layers:
            # Concatenate attributes before each ConvTranspose2d layer
            x = torch.cat([x, attributes], dim=1)
            x = layer(x)  # Pass through the sequence (ConvTranspose2d + BatchNorm + Activation)

        return x

class AutoEncoder(nn.Module):
    """
    Full AutoEncoder: Combines the Encoder and Decoder.
    """
    def __init__(self, n_attributes):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()  # Use the defined Encoder
        self.decoder = Decoder(n_attributes)  # Use the defined Decoder and specify the number of attributes

    def forward(self, x, attributes):
        # Encoder compresses the input image
        latent = self.encoder(x)  # Output shape: [batch_size, 512, h, w]
        # Directly pass the attribute vector to the decoder
        output = self.decoder(latent, attributes)  # Decoder handles attribute expansion internally
        return output

class Discriminator(nn.Module):
    """
    Discriminator: Combines convolutional layers and fully connected layers into a single Sequential module.
    Includes Dropout as per the paper's requirements.
    """
    def __init__(self, n_attributes):
        super(Discriminator, self).__init__()
        self.n_attributes = n_attributes

        # Combine convolutional and fully connected layers
        self.model = nn.Sequential(
            # Convolutional layer (C_512)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # C_512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Dropout applied with a rate of 0.3

            # Flatten layer
            nn.Flatten(),  # Automatically flatten to [batch_size, 512 * 2 * 2]

            # Fully connected layers
            nn.Linear(2048, 512),  # First fully connected layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Dropout applied with a rate of 0.3
            nn.Linear(512, n_attributes),  # Second fully connected layer
            nn.Sigmoid()  # Normalize outputs to [0, 1]
        )

    def forward(self, x):
        """
        Forward pass of the Discriminator.

        Parameters:
        z: Latent representation (shape: [batch_size, 512, 4, 4])

        Returns:
        - attributes: Predicted attributes (shape: [batch_size, n_attributes])
        """
        # Pass through the entire model
        x = self.model(x)
        return x

