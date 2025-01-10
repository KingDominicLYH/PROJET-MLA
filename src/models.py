import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            # Layer 1: Input channels = 3 (RGB), Output channels = 16
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 4: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 5: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 6: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 7: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.n_attributes = params.n_attributes
        self.attribute_channels = 2 * self.n_attributes

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512 + self.attribute_channels, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512 + self.attribute_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256 + self.attribute_channels, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128 + self.attribute_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64 + self.attribute_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32 + self.attribute_channels, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(16 + self.attribute_channels, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        ])

    def forward(self, z, attributes):
    # Extract the batch size and spatial dimensions (height, width) from z
    # batch_size: number of images in a batch
    # _: the channel dimension (not used explicitly here)
    # h, w: current spatial height and width of z
    batch_size, _, h, w = z.shape

    # Reshape attributes from [batch_size, n_attributes] to [batch_size, n_attributes*2, 1, 1]
    # so it can be broadcast (expanded) across the spatial dimensions.
    # 'n_attributes * 2' corresponds to the one-hot representation (e.g., [1,0] or [0,1]) for each attribute.
    attributes = attributes.reshape(batch_size, -1, 1, 1)

    # Expand attributes along the height and width dimensions so that they match (h, w) of z.
    # The result will have shape [batch_size, n_attributes*2, h, w].
    attributes = attributes.expand(batch_size, -1, h, w)

    # Initialize x with the latent representation z before proceeding through the decoder layers.
    x = z

    # Iterate through each transposed convolution layer in the decoder.
    for layer in self.layers:
        # Get the current spatial size of x to know how to resize the attributes.
        _, _, h, w = x.shape

        # Resize the attribute tensor to the current spatial dimensions (h, w) of x
        # (if those dimensions have changed from the previous layer).
        attributes_resized = nn.functional.interpolate(attributes, size=(h, w), mode="nearest")

        # Concatenate the resized attributes along the channel dimension (dim=1).
        # This provides attribute information to the decoder at every layer.
        x = torch.cat([x, attributes_resized], dim=1)

        # Pass the concatenated tensor through the current layer (ConvTranspose2d + optional BatchNorm + ReLU/Tanh).
        x = layer(x)

    # Return the final output, which should be a reconstructed image with channels = 3 (RGB).
    return x

class AutoEncoder(nn.Module):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.params = params
        self.encoder = Encoder(params)  # Use the defined Encoder
        self.decoder = Decoder(params)  # Use the defined Decoder and specify the number of attributes

    def forward(self, x, attributes):
        # Encoder compresses the input image
        latent = self.encoder(x)  # Output shape: [batch_size, 512, h, w]
        # Directly pass the attribute vector to the decoder
        output = self.decoder(latent, attributes)  # Decoder handles attribute expansion internally
        return output

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.params = params
        self.n_attributes = params.n_attributes

        # Combine convolutional and fully connected layers
        self.model = nn.Sequential(
            # Convolutional layer (C_512)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # C_512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten layer
            nn.Flatten(),  # Automatically flatten to [batch_size, 512 * 1 * 1]

            # Fully connected layers
            nn.Linear(512, 512),  # First fully connected layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Dropout applied with a rate of 0.3
            nn.Linear(512, params.n_attributes*2),  # Second fully connected layer
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
        # Reshape output to [batch_size, n_attributes, 2] for One-Hot encoding
        x = x.view(x.size(0), self.n_attributes, 2)

        # Apply softmax along the last dimension to get probabilities
        x = torch.softmax(x, dim=-1)
        return x

class Classifier(nn.Module):
    """
    The Classifier predicts attribute values directly from an input image.

    This network consists of multiple convolutional layers followed by a fully 
    connected block. The output shape is [batch_size, n_attributes, 2], indicating 
    a binary classification for each attribute.
    """
    def __init__(self, params):

        super(Classifier, self).__init__()

        # Convolutional layers (8 layers in total), each halving the spatial resolution
        # and increasing or preserving the number of feature channels.
        self.conv_layers = nn.Sequential(
            # Layer 1: 3 -> 16
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 4: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 5: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 6: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 7: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Layer 8: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Flatten operation to prepare for the fully connected layers
        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),  # Connect the flattened features (512) to a 512-dimensional layer
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),              # Dropout for regularization
            nn.Linear(512, params.n_attributes * 2),  # 2 outputs (binary classification) per attribute
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x.view(x.size(0), -1, 2)
