import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("model")

class DoubleConv2D(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv2D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down2D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up2D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up2D, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Adjust input channels for DoubleConv2D after concatenation
            self.conv = DoubleConv2D(in_channels, out_channels) # Input channels = skip connection + upsampled
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Adjust input channels for DoubleConv2D after concatenation
            self.conv = DoubleConv2D(in_channels, out_channels) # Input channels = skip connection + upsampled (in_channels // 2 + in_channels // 2 = in_channels)


    def forward(self, x1, x2):
        # x1 is the output from the previous layer (upsampled)
        # x2 is the skip connection from the corresponding downsampling layer
        x1 = self.up(x1)

        # input is BCHW (batch, channel, height, width)
        # Pad x1 to match the spatial dimensions of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Apply padding: (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, bilinear=False, initial_filters=64): # Increased initial filters for 2D
        """
        Implementation of 2D U-Net

        Args:
            n_channels: Number of input channels (4 for BraTS - T1, T1c, T2, FLAIR for a single slice)
            n_classes: Number of output classes (4 for BraTS - background, edema, enhancing tumor, non-enhancing tumor)
            bilinear: Whether to use bilinear upsampling
            initial_filters: Number of filters in first layer (will double in each down step)
        """
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        f = initial_filters  # Base number of filters

        self.inc = DoubleConv2D(n_channels, f)
        self.down1 = Down2D(f, 2*f)
        self.down2 = Down2D(2*f, 4*f)
        self.down3 = Down2D(4*f, 8*f)
        # Determine the factor for the bottleneck channels based on bilinear setting
        factor = 2 if bilinear else 1
        self.down4 = Down2D(8*f, 16*f // factor) # Bottleneck layer

        # Adjust channel sizes for Up layers based on concatenation
        # Up(in_channels, out_channels)
        # in_channels = channels from skip connection + channels from upsampled layer
        self.up1 = Up2D(16*f, 8*f // factor, bilinear) # Input: 8f (skip) + 8f/factor (upsampled) = 16f/factor if not bilinear else 8f+8f=16f
        self.up2 = Up2D(8*f, 4*f // factor, bilinear)  # Input: 4f (skip) + 4f/factor (upsampled)
        self.up3 = Up2D(4*f, 2*f // factor, bilinear)  # Input: 2f (skip) + 2f/factor (upsampled)
        self.up4 = Up2D(2*f, f, bilinear)             # Input: f (skip) + f (upsampled)

        self.outc = OutConv2D(f, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # Initial convolution block
        x2 = self.down1(x1)   # First down-sampling block
        x3 = self.down2(x2)   # Second down-sampling block
        x4 = self.down3(x3)   # Third down-sampling block
        x5 = self.down4(x4)   # Fourth down-sampling block (bottleneck)

        # Upsampling path with skip connections
        x = self.up1(x5, x4)  # First up-sampling block
        x = self.up2(x, x3)  # Second up-sampling block
        x = self.up3(x, x2)  # Third up-sampling block
        x = self.up4(x, x1)  # Fourth up-sampling block

        logits = self.outc(x) # Final output convolution
        return logits

    def get_model_parameters(self):
        return sum(p.numel() for p in self.parameters())

# Test the model
if __name__ == "__main__":
    # Create small test input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate 2D U-Net
    model = UNet2D(n_channels=4, n_classes=4).to(device)
    logger.info(f"Model created with {model.get_model_parameters()} parameters")

    # Test with a batch of 2D slices (e.g., batch_size=2, channels=4, height=128, width=128)
    x = torch.randn(2, 4, 128, 128).to(device)
    logger.info(f"Input shape: {x.shape}") # Should be [B, C, H, W]

    y = model(x)
    logger.info(f"Output shape: {y.shape}") # Should be [B, n_classes, H, W]
