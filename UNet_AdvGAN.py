# This U-Net AdvGAN uses the Vanilla Discriminator.

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_Generator(nn.Module):
  def __init__(self, image_channels):
    super().__init__()

    self.image_channels = image_channels

  # Accepts image tensors of 1x28x28
  def encoding_block(self, x : torch.tensor, input_channels : int, output_channels : int):
    block = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    return block(x)

  def decoding_block(self, x : torch.tensor, input_channels : int, output_channels : int, stride : int = 2):
    middle_channels = output_channels * 2
    block = nn.Sequential(
        nn.Conv2d(input_channels, middle_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(middle_channels, output_channels, kernel_size=2, stride=stride) # using stride=2 here upsizes the image
    )
    return block(x)

  def output_block(self, x : torch.tensor, input_channels : int, output_channels : int):
    middle_channels = input_channels // 2
    block = nn.Sequential(
        nn.Conv2d(input_channels, middle_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(middle_channels, output_channels, kernel_size=1, stride=1, padding=0),
        nn.Conv2d(output_channels, output_channels, kernel_size=18, stride=6, padding=0) # DOWNSIZING LAYER; kernel, stride, and padding set for output of [1, 28, 28] (the numbers are so round :') )
    )
    return block(x)

  def forward(self, x, device : str = "cpu"):
    x = x.to(device)
    x = F.interpolate(x, scale_factor=20, mode="nearest-exact")

    # Encoder
    e1 = self.encoding_block(x, self.image_channels, 64).to(device) # [image_channels, 560, 560] -> [64, 279, 279]
    e2 = self.encoding_block(e1, 64, 128).to(device)  # [64, 279, 279] -> [128, 138, 138]
    e3 = self.encoding_block(e2, 128, 256).to(device)  # [128, 138, 138] -> [256, 68, 68]
    e4 = self.encoding_block(e3, 256, 512).to(device)  # [256, 68, 68] -> [512, 33, 33]

    # Decoder
    d1 = self.decoding_block(e4, 512, 512, 1).to(device)  # [512, 33, 33] -> [512, 30, 30]

    e4 = F.interpolate(e4, size=[30, 30], mode="bilinear") # e4: [512, 33, 33] -> [512, 30, 30]
    e4d1 = torch.cat([e4, d1], dim=1) # [512, 30, 30] + [512, 30, 30] -> [1024, 30, 30]
    d2 = self.decoding_block(e4d1, 1024, 256).to(device) # [1024, 30, 30] -> [256, 52, 52]

    e3 = F.interpolate(e3, size=[52, 52], mode="bilinear") # [256, 68, 68] -> [256, 52, 52]
    e3d2 = torch.cat([e3, d2], dim=1) # [256, 52, 52] + [256, 52, 52] = [512, 52, 52]
    d3 = self.decoding_block(e3d2, 512, 128).to(device) # [512, 52, 52] -> [128, 96, 96]

    e2 = F.interpolate(e2, size=[96, 96], mode="bilinear") # [128, 138, 138] -> [128, 96, 96]
    e2d3 = torch.cat([e2, d3], dim=1) # [128, 96, 96] + [128, 96, 96] = [256, 96, 96]
    d4 = self.decoding_block(e2d3, 256, 64).to(device) # [256, 96, 96] -> [64, 184, 184]

    # Output layer
    e1 = F.interpolate(e1, size=[184, 184], mode="bilinear") # [64, 279, 279] -> [64, 184, 184]
    e1d4 = torch.cat([e1, d4], dim=1) # [64, 184, 184] + [64, 184, 184] -> [128, 184, 184]
    perturbation = self.output_block(e1d4, 128, self.image_channels).to(device) # [128, 184, 184] -> [image_channels, 28, 28]
    return perturbation