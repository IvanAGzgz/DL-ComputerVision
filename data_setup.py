"""
Contiene la funcionalidad para crear DataLoaders de PyTorch 
para datos de clasificación de imágenes.
"""

import os
import torch
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Crea DataLoaders para entrenamiento y prueba.

  Recibe la ruta de un directorio de entrenamiento y de un directorio de prueba,
  los convierte en conjuntos de datos (Datasets) de PyTorch y luego en 
  DataLoaders de PyTorch.

  Argumentos:
    train_dir: Ruta al directorio de train.
    test_dir: Ruta al directorio de test.
    transform: Transformaciones de torchvision para aplicar a los datos de train y test.
    batch_size: Número de muestras por batch en cada uno de los DataLoaders.
    num_workers: Un número entero que indica el número de procesos (workers) por DataLoader.

  Retorna:
    Una tupla de (train_dataloader, test_dataloader, class_names).
    Donde class_names es una lista con las clases objetivo.
    Ejemplo de uso:
      train_dataloader, test_dataloader, class_names = \
        create_dataloaders(train_dir=ruta/a/train_dir,
                           test_dir=ruta/a/test_dir,
                           transform=alguna_transformacion,
                           batch_size=32,
                           num_workers=4)
  """
  # Usa ImageFolder para crear los datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Obtiene los nombres de las clases
  class_names = train_data.classes

  # Convierte las imágenes en DataLoaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,  # mezcla aleatoriamente los datos de entrenamiento
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,  # no es necesario mezclar los datos de prueba
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
