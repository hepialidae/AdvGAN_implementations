import torch
from tqdm import tqdm
from pathlib import Path

def train_model(model, train_dataset, test_dataset, loss_fn : torch.nn, optim_fn : torch.optim, epochs : int, save_folder : Path, save_file_name : str, device : str):
  save_path = save_folder / save_file_name

  # If a state_dict exists, load it
  if save_path.exists():
    model.load_state_dict(torch.load(save_path, map_location=torch.device(device)))
    model = model.to(device)
  else:
    for epoch in range(epochs):
      model.train()
      train_loss, test_loss = 0, 0
      for j, (train_imgs, train_labels) in enumerate(tqdm(train_dataset, desc=f"Target Model Training | Epoch: {epoch + 1}/{epochs}", leave=True)):
        train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

        train_pred = model(train_imgs)

        loss = loss_fn(train_pred, train_labels)
        train_loss += loss.item()

        optim_fn.zero_grad()

        loss.backward()

        optim_fn.step()

      model.eval()
      with torch.inference_mode():
        for k, (test_imgs, test_labels) in enumerate(tqdm(test_dataset, desc=f"Target Model Testing | Epoch: {epoch + 1}/{epochs}", leave=True)):
          test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
          test_pred = model(test_imgs)

          loss = loss_fn(test_pred, test_labels)
          test_loss += loss.item()

      print(f"\nEpoch: {epoch + 1} | Train Loss: {train_loss / len(train_dataset)} | Test Loss: {test_loss / len(test_dataset)}\n")
    save_folder.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model.state_dict(), f=save_path)

# Classifies an 1x28x28 image tensor as digit 0-9
def inference_target_model(model, input, device):
  input = input.to(device)
  logits = model(input)
  return torch.argmax(torch.softmax(logits, dim=1), dim=1)