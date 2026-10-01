import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

from target_model_architecture import MNIST_classifier
from train_target_model import train_model
from vanilla_AdvGAN_architecture import Discriminator, Generator
from train_AdvGAN import AdvGAN_Attack

if __name__ == "__main__":
    print("Imports completed.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Load datasets into DataLoaders
    # Settings:
    batch_size = 32
    num_workers = 1

    print("Loading datasets...")
    train_mnist = MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
    test_mnist = MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_mnist, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ## Create and train undefended MNIST classifier
    print("Classifier instantiated.")
    classifier = MNIST_classifier(input_shape=1, hidden_units=32, output_shape=10).to(device)

    # Settings:
    epochs = 5
    target_loss_fn = nn.CrossEntropyLoss()
    target_optim_fn = torch.optim.Adam(params=classifier.parameters(), lr=0.0003)
    model_folder = Path("models") # for target model and GAN
    model_name = "target_model_statedict.pth"
    discriminator_name = "discriminator_statedict.pth"
    generator_name = "generator_statedict.pth"

    train_model(classifier, train_dataloader, test_dataloader, target_loss_fn, target_optim_fn, epochs, model_folder, model_name, device)

    print("Instantiating AdvGAN...")
    vanilla_discriminator = Discriminator(1)
    vanilla_generator = Generator(1, 128)

    vanilla_AdvGAN = AdvGAN_Attack(vanilla_discriminator, vanilla_generator, classifier, 10, nn.CrossEntropyLoss(reduction="mean"), nn.BCEWithLogitsLoss(reduction="mean"), torch.optim.Adam(params=vanilla_discriminator.parameters(), lr=0.0003), torch.optim.Adam(params=vanilla_generator.parameters(), lr=0.0003),
                               generator_loss_alpha=1.0, generator_loss_beta=1.0, kappa=0.1, c=0.5, batch_size=32, device=device)
    vanilla_AdvGAN.train_advgan(train_dataloader, test_dataloader,
                            epochs=15, 
                            save_folder=model_folder, discriminator_file_name=discriminator_name, generator_file_name=generator_name)