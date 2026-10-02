import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Subset
from pathlib import Path

# The following function aims to calculate the loss of the generator using the loss formulas described in the original AdvGAN paper.
# Paper: https://arxiv.org/abs/1801.02610$0

class AdvGAN_Attack:
  def __init__(self, discriminator, generator, target_model, num_target_labels : int, target_loss_fn,
               D_loss_fn, D_optim_fn, G_optim_fn, # D = discriminator, G = generator; G_loss_fn uses get_generator_loss by default
               generator_loss_alpha : int | float, # alpha of the loss function as described in the AdvGAN paper
               generator_loss_beta : int | float, # beta of the loss function as described in the AdvGAN paper
               kappa : int | float, # minimum confidence in false prediction as described by C&W
               c : int | float, # maximum allowed perturbation
               batch_size : int | float, device):
    self.batch_size = batch_size
    self.device = device

    self.discriminator = discriminator.to(device)
    self.generator = generator.to(device)

    self.target_model = target_model.to(device)
    self.num_target_labels = num_target_labels
    self.target_loss_fn = target_loss_fn

    self.D_loss_fn = D_loss_fn
    self.D_optim_fn = D_optim_fn
    self.G_optim_fn = G_optim_fn
    self.generator_loss_alpha = generator_loss_alpha
    self.generator_loss_beta = generator_loss_beta
    self.kappa = kappa
    self.c = c

  def get_generator_loss(self, x : torch.tensor, # original images
                       y : torch.tensor, # labels for x
                       dx : torch.tensor, # the result of the discriminator when given x
                       gx : torch.tensor, # the result of the generator when given x
                       targeted : bool = False, target : torch.IntTensor | torch.LongTensor = None): # settings for targeted attacks and non-targeted attacks
    x, y, dx, gx = x.to(self.device), y.to(self.device), dx.to(self.device), gx.to(self.device)

    Lgan = torch.mean(torch.log(dx)) + torch.mean(torch.log(1 - self.discriminator(x + gx))) # torch.mean is used to calculate expected value

    # My implementation varies here: nn.CrossEntropyLoss only accepts model logits, so I use f(x + G(x)) instead of x + G(x)
    if targeted and target != None:
      Lf_adv = torch.mean(self.target_loss_fn(self.target_model(x + gx), target))
    else:
      # C&W untargeted loss
      logits = self.target_model(x + gx)
      batch_indices = torch.arange(logits.size(0), device=self.device)
      correct_logits = logits[batch_indices, y]

      masked_logits = logits.clone()
      masked_logits[batch_indices, y] = float('-inf')
      max_incorrect_logits, _ = torch.max(masked_logits, dim=1)

      Lf_adv = torch.mean(torch.maximum(correct_logits - max_incorrect_logits, torch.tensor(-1 * self.kappa)))

    # An alternate way of implementing max(0, ||G(x)|| - c)
    Lhinge = torch.mean(torch.clamp(torch.linalg.vector_norm(gx), min=0, max=self.c))

    total_loss = Lf_adv + self.generator_loss_alpha * Lgan + self.generator_loss_beta * Lhinge

    # MAE to find the difference between total_loss and 0, avoiding negative loss
    final_loss = F.l1_loss(total_loss, torch.zeros_like(total_loss), reduction="mean")
    return final_loss

  def train_or_test_advgan(self, img_batch, label_batch, train : bool = True,
                           targeted : bool = False, target : torch.IntTensor | torch.LongTensor = None):
    img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)

    perturbation = self.generator(img_batch)
    adv_imgs = img_batch + perturbation
    adv_imgs = adv_imgs.to(self.device)

    # Optimize the discriminator
    # This implementation treats ouputs of 1 as real and outputs of 0 as fake.
    D_pred_real = self.discriminator(img_batch)
    D_pred_fake = self.discriminator(adv_imgs)

    D_loss_real = self.D_loss_fn(D_pred_real, torch.ones_like(D_pred_real))
    D_loss_fake = self.D_loss_fn(D_pred_fake, torch.zeros_like(D_pred_fake))

    if train:
      self.D_optim_fn.zero_grad()

      D_loss_real.backward()
      D_loss_fake.backward()

      self.D_optim_fn.step()

    D_loss = D_loss_real + D_loss_fake

    # Optimize the generator
    G_loss = self.get_generator_loss(img_batch, label_batch, self.discriminator(img_batch), self.generator(img_batch), targeted, target)

    if train:
      self.G_optim_fn.zero_grad()

      G_loss.backward()

      self.G_optim_fn.step()

    return D_loss.item(), G_loss.item()

  def train_advgan(self, train_dataloader, test_dataloader, targeted : bool = False,
                   train_target : torch.IntTensor | torch.LongTensor = None, # train_target should be a tensor containing all of the targets, corresponding to the training Dataloader
                   test_target : torch.IntTensor | torch.LongTensor = None,
                   epochs : int = 1, 
                   save_folder : Path = Path("models"), discriminator_file_name : str = "discriminator_statedict.pth", generator_file_name : str = "generator_statedict.pth"):
    discriminator_save_path = save_folder / discriminator_file_name
    generator_save_path = save_folder / generator_file_name

    if discriminator_save_path.exists() or generator_save_path.exists():
      self.discriminator.load_state_dict(torch.load(discriminator_save_path, map_location=torch.device(self.device)))
      self.discriminator = self.discriminator.to(self.device)
      self.generator.load_state_dict(torch.load(generator_save_path, map_location=torch.device(self.device)))
      self.generator = self.generator.to(self.device)
    else:
      if targeted:
        train_target, test_target = train_target.to(self.device), test_target.to(self.device)

      train_accs = []
      test_accs = []

      self.target_model.eval()

      for epoch in range(epochs):
        # Training
        self.discriminator.train()
        self.generator.train()

        D_train_loss, G_train_loss, target_train_acc = 0, 0, 0

        for i, (imgs, labels) in enumerate(tqdm(train_dataloader, desc=f"AdvGAN Training | Epoch: {epoch + 1}/{epochs}")):
          imgs, labels = imgs.to(self.device), labels.to(self.device)

          target_subset = Subset(train_target, range(i * self.batch_size, (i + 1) * self.batch_size))
          D_loss, G_loss = self.train_or_test_advgan(imgs, labels, train=True, targeted=targeted, target=target_subset)

          D_train_loss += D_loss
          G_train_loss += G_loss

          attack_logits = self.target_model(imgs + self.generator(imgs))
          attack_pred = torch.argmax(torch.softmax(attack_logits, dim=1), dim=1)
          target_train_acc += (attack_pred == labels).sum().item()/len(attack_pred)

        # Testing
        self.discriminator.eval()
        self.generator.eval()
        with torch.inference_mode():
          D_test_loss, G_test_loss, target_test_acc = 0, 0, 0

          for j, (imgs, labels) in enumerate(tqdm(test_dataloader, desc=f"AdvGAN Testing | Epoch: {epoch + 1}/{epochs}")):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            target_subset = Subset(test_target, range(j * self.batch_size, (j + 1) * self.batch_size))
            D_loss, G_loss = self.train_or_test_advgan(imgs, labels, train=False, targeted=targeted, target=target_subset)

            D_test_loss += D_loss
            G_test_loss += G_loss

            # Attack success rate because loss isn't really a reliable indicator of performance
            attack_logits = self.target_model(imgs + self.generator(imgs))
            attack_pred = torch.argmax(torch.softmax(attack_logits, dim=1), dim=1)
            target_test_acc += (attack_pred == labels).sum().item()/len(attack_pred)

        print(f"\nEpoch: {epoch + 1}")
        print(f"D_train_loss: {D_train_loss/len(train_dataloader):.3f} | G_train_loss: {G_train_loss/len(train_dataloader):.3f}")
        print(f"D_test_loss: {D_test_loss/len(test_dataloader):.3f} | G_test_loss: {G_test_loss/len(test_dataloader):.3f}")

        train_acc = target_train_acc / len(train_dataloader) * 100
        train_accs.append(train_acc)
        test_acc = target_test_acc / len(test_dataloader) * 100
        test_accs.append(test_acc)
        print(f"Target Model Accuracy (Train): {train_acc:.2f}%")
        print(f"Target Model Accuracy (Test): {test_acc:.2f}%")
        print(f"\n")

      # Output results
      plt.figure(figsize=[4, 4])
      plt.plot(train_accs)
      plt.title("Target Model Training Accuracy Over Time")

      plt.figure(figsize=[4, 4])
      plt.plot(test_accs)
      plt.title("Target Model Testing Accuracy Over Time")

      # Save model
      save_folder.mkdir(parents=True, exist_ok=True)
      torch.save(obj=self.discriminator.state_dict(), f=discriminator_save_path)
      torch.save(obj=self.discriminator.state_dict(), f=generator_save_path)

# Takes in a 1x28x28 image tensor and returns a adversarial example of the same shape
  def inference_AdvGAN(self, input):
    input = input.to(self.device)
    return input + self.generator(input)