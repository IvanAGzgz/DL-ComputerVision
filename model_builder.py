"""
Contiene el código del modelo de PyTorch para instanciar un modelo TinyVGG.
"""
import torch
from torch import nn 

class TinyVGG(nn.Module):
  """Crea la arquitectura de TinyVGG.

  Replica la arquitectura de TinyVGG del sitio web de CNN explainer en PyTorch.
  Ver la arquitectura original aquí: https://poloclub.github.io/cnn-explainer/

  Argumentos:
    input_shape: Un entero que indica el número de canales de entrada.
    hidden_units: Un entero que indica el número de unidades ocultas entre capas.
    output_shape: Un entero que indica el número de unidades de salida.
  """

  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # ¿De dónde proviene esta forma de in_features? 
          # Es porque cada capa de nuestra red comprime y cambia la forma de los datos de entrada.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # Devuelve self.classifier(self.conv_block_2(self.conv_block_1(x))) 
