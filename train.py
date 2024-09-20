"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hiperparametros

if __name__ == "__main__":

  # Editable    
  NUM_EPOCHS = 5
  BATCH_SIZE = 32
  HIDDEN_UNITS = 128
  LEARNING_RATE = 0.001

  # Setup directorios
  train_dir = "data/cancer_data/train"
  test_dir = "data/cancer_data/test"

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Crear transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Crear DataLoaders mediante data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  # Crear modelo mediante model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=HIDDEN_UNITS,
      output_shape=len(class_names)
  ).to(device)

  # Configurar loss y optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Training mediante engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)

  # Guardar modelo mediante utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="model.pth")