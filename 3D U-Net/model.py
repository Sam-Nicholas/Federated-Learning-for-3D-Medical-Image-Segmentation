import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("model")

class DoubleConv3D(nn.Module):
    """Applies two consecutive 3D convolutions with BatchNorm and ReLU activation."""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of channels in the intermediate layer.
                                         Defaults to out_channels if None.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            padding (int, optional): Padding for convolutions. Defaults to 1.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # First convolution block
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False), # Bias often disabled when using BatchNorm
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            # Second convolution block
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Passes input through the double convolution block."""
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling block: MaxPool3D followed by a DoubleConv3D block."""
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for the DoubleConv3D block.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2), # Halve spatial dimensions (D, H, W)
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        """Applies max pooling and then the double convolution."""
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling block: Upsample/ConvTranspose3D followed by DoubleConv3D, includes skip connection."""
    def __init__(self, in_channels, out_channels, bilinear=False):
        """
        Args:
            in_channels (int): Number of channels from the upsampled feature map.
            out_channels (int): Number of output channels for the DoubleConv3D block.
            bilinear (bool, optional): If True, use bilinear upsampling followed by a 1x1 conv.
                                       If False, use ConvTranspose3d. Defaults to False.
        """
        super().__init__()

        # Determine the number of input channels to the final DoubleConv3D block
        # It will be the sum of channels from the upsampled path and the skip connection
        # The skip connection comes from the corresponding Down block, which has 'out_channels' from this Up block's perspective
        conv_in_channels = out_channels + (in_channels // 2 if not bilinear else in_channels)


        if bilinear:
             # Bilinear upsampling doesn't learn parameters, often followed by a conv to adjust channels
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # The subsequent DoubleConv handles the channel reduction and feature fusion
            # Note: The DoubleConv input channels need careful calculation based on concatenation
            self.conv = DoubleConv3D(in_channels + out_channels, out_channels) # Input: skip_connection + upsampled
        else:
            # Transposed convolution learns parameters for upsampling
            # It typically halves the number of channels before concatenation
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
             # Input channels = channels from skip connection + channels after ConvTranspose3d
            self.conv = DoubleConv3D(out_channels + in_channels // 2, out_channels)


    def forward(self, x1, x2):
        """
        Performs the upsampling, concatenation with skip connection, and convolution.

        Args:
            x1 (torch.Tensor): Feature map from the previous layer (to be upsampled).
            x2 (torch.Tensor): Feature map from the corresponding encoder layer (skip connection).

        Returns:
            torch.Tensor: Output feature map.
        """
        x1 = self.up(x1) # Upsample x1 to match spatial dimensions of x2

        # Handle potential size mismatches due to odd dimensions / padding choices in the encoder
        # Input tensor format: (Batch, Channel, Depth, Height, Width)
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        # Pad x1 if necessary to match x2's spatial dimensions for concatenation
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2, # Pad width
                        diffH // 2, diffH - diffH // 2, # Pad height
                        diffD // 2, diffD - diffD // 2]) # Pad depth

        # Concatenate the upsampled feature map (x1) with the skip connection (x2) along the channel dimension
        x = torch.cat([x2, x1], dim=1)

        # Apply the double convolution block to the concatenated features
        return self.conv(x)

class OutConv3D(nn.Module):
    """Final 1x1x1 convolution layer to map feature channels to the number of output classes."""
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output classes.
        """
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Applies the 1x1x1 convolution."""
        return self.conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net architecture for volumetric segmentation.

    Reference: Özgün Çiçek et al. "3D U-Net: Learning Dense Volumetric Segmentation from Scarce Annotation"
               (https://arxiv.org/abs/1606.06650)
    """
    def __init__(self, n_channels=4, n_classes=4, bilinear=False, initial_filters=16):
        """
        Initialises the 3D U-Net model.

        Args:
            n_channels (int, optional): Number of input channels (e.g., 4 for BraTS modalities T1, T1c, T2, FLAIR). Defaults to 4.
            n_classes (int, optional): Number of output classes (e.g., 4 for BraTS background, oedema, non-enhancing, enhancing). Defaults to 4.
            bilinear (bool, optional): Whether to use bilinear upsampling in the decoder path. Defaults to False (uses ConvTranspose3d).
            initial_filters (int, optional): Number of filters in the first convolutional layer. This number typically doubles at each downsampling step. Defaults to 16.
        """
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        f = initial_filters # Base number of filters

        # --- Encoder Path ---
        self.inc = DoubleConv3D(n_channels, f)      # Initial convolution block
        self.down1 = Down3D(f, 2*f)                 # Downsample 1
        self.down2 = Down3D(2*f, 4*f)               # Downsample 2
        self.down3 = Down3D(4*f, 8*f)               # Downsample 3
        self.down4 = Down3D(8*f, 16*f)              # Downsample 4 (Bottleneck)

        # --- Decoder Path ---
        # Note: Input channels for Up blocks depend on the output channels of the layer below
        # and the corresponding skip connection layer.
        self.up1 = Up3D(16*f, 8*f, bilinear)        # Upsample 1 + Skip connection from down3
        self.up2 = Up3D(8*f, 4*f, bilinear)         # Upsample 2 + Skip connection from down2
        self.up3 = Up3D(4*f, 2*f, bilinear)         # Upsample 3 + Skip connection from down1
        self.up4 = Up3D(2*f, f, bilinear)           # Upsample 4 + Skip connection from inc

        # --- Output Layer ---
        self.outc = OutConv3D(f, n_classes)         # Final 1x1x1 convolution

        logger.info(f"Initialised 3D U-Net with initial_filters={initial_filters}, bilinear={bilinear}")

    def forward(self, x):
        """
        Defines the forward pass of the 3D U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Depth, Height, Width).

        Returns:
            torch.Tensor: Output tensor (logits) of shape (Batch, Classes, Depth, Height, Width).
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck features

        # Decoder with skip connections
        x = self.up1(x5, x4) # Upsample x5, concatenate with x4
        x = self.up2(x, x3)  # Upsample result, concatenate with x3
        x = self.up3(x, x2)  # Upsample result, concatenate with x2
        x = self.up4(x, x1)  # Upsample result, concatenate with x1

        # Output convolution
        logits = self.outc(x)
        return logits

    def get_model_parameters(self):
        """Calculates and returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --- Model Test ---
if __name__ == "__main__":
    # Basic test to check model instantiation and forward pass shapes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing model on device: {device}")

    # Instantiate the model
    test_model = UNet3D(n_channels=4, n_classes=4, initial_filters=16).to(device)
    logger.info(f"Model instantiated with {test_model.get_model_parameters():,} parameters.")

    # Create a dummy input tensor (Batch=1, Channels=4, Depth=64, Height=64, Width=64)
    dummy_input = torch.randn(1, 4, 64, 64, 64).to(device)
    logger.info(f"Input tensor shape: {dummy_input.shape}")

    # Perform a forward pass
    with torch.no_grad(): # Disable gradient calculation for testing
        output_logits = test_model(dummy_input)
    logger.info(f"Output tensor shape: {output_logits.shape}")

    # Verify output shape matches expectation (Batch, Classes, Depth, Height, Width)
    assert output_logits.shape == (1, 4, 64, 64, 64), "Output shape mismatch!"
    logger.info("Model forward pass successful.")
