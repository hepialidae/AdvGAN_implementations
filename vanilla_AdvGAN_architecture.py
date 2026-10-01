import torch.nn as nn

# Modified from GeeksForGeeks's Vanilla GAN Implementation:
# https://www.geeksforgeeks.org/deep-learning/generative-adversarial-network-gan/$0
# Originally, the VanillaGAN was made to generate images based on noise, for the CIFAR dataset.
# I updated its input dimensions and edited its generator to produce perturbations instead of images.

# This implementation treats ouputs of 1 as real and outputs of 0 as fake.
class Discriminator(nn.Module):
  def __init__(self, image_channels):
    super(Discriminator, self).__init__()

    self.image_channels = image_channels

    self.layer1 = nn.Sequential( # [batch_size=1, 1, 28, 28] -> [1, 32, 14, 14]
      nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.25)
    )

    self.layer2 = nn.Sequential( # [1, 32, 14, 14] -> [1, 64, 7, 7]
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    )

    self.layer3 = nn.Sequential( # [1, 64, 7, 7] -> [1, 64, 8, 8]
      nn.ZeroPad2d((0, 1, 0, 1)),
      nn.BatchNorm2d(64, momentum=0.82),
      nn.LeakyReLU(0.25),
      nn.Dropout(0.25)
    )

    self.layer4 = nn.Sequential( # [1, 64, 8, 8] --> [1, 128, 4, 4]
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(128, momentum=0.82),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.25),
    )

    self.layer5 = nn.Sequential( # [1, 128, 4, 4] -> [1, 256, 4, 4]
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256, momentum=0.8),
      nn.LeakyReLU(0.25),
      nn.Dropout(0.25)
    )

    self.layer6 = nn.Sequential( # [1, 256, 4, 4] -> [1, 4096]
      nn.Flatten()
    )

    self.layer7 = nn.Sequential( # [1, 4096] -> [1, 1]
      nn.Linear(4096, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.layer1(x)
    output = self.layer2(output)
    output = self.layer3(output)
    output = self.layer4(output)
    output = self.layer5(output)
    output = self.layer6(output)
    output = self.layer7(output)
    return output

class Generator(nn.Module):
  def __init__(self, image_channels, hidden_units):
    super(Generator, self).__init__()

    self.image_channels = image_channels
    self.hidden_units = hidden_units

    self.block1 = nn.Sequential(
      nn.Conv2d(image_channels, hidden_units, kernel_size=3, padding=1),  # [batch_size=1, 1, 28, 28] -> [1, 128, 28, 28]
      nn.BatchNorm2d(hidden_units, momentum=0.78),
      nn.ReLU()
    )

    self.block2 = nn.Sequential(
      nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),    # [1, 128, 28, 28] -> [1, 128, 28, 28]
      nn.BatchNorm2d(hidden_units, momentum=0.78),
      nn.ReLU()
    )

    self.block3 = nn.Sequential(
      nn.Conv2d(hidden_units, image_channels, kernel_size=3, padding=1),  # [1, 128, 28, 28] -> [1, 1, 28, 28]
      nn.Tanh()
    )

  def forward(self, x):
    perturbation = self.block1(x)
    perturbation = self.block2(perturbation)
    perturbation = self.block3(perturbation)
    return perturbation
