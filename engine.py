"""
Contiene funciones para entrenar y probar un modelo de PyTorch.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Entrena un modelo de PyTorch por un epoch.

  Cambia un modelo target de PyTorch al modo de train y luego
  ejecuta todos los pasos de train requeridos (paso hacia adelante,
  cálculo de la pérdida, paso del optimizador).

  Argumentos:
    model: Un modelo de PyTorch a ser entrenado.
    dataloader: Una instancia de DataLoader para entrenar el modelo.
    loss_fn: Una función de pérdida de PyTorch a minimizar.
    optimizer: Un optimizador de PyTorch para ayudar a minimizar la función de pérdida.
    device: El dispositivo en el que se computará (e.g. "cuda" o "cpu").

  Retorna:
    Una tupla con los valores de pérdida y precisión del train.
    En la forma (train_loss, train_accuracy). Por ejemplo:

    (0.1112, 0.8743)
  """

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Test de un modelo de PyTorch por epoch.

  Cambia un modelo objetivo de PyTorch al modo "eval" y luego realiza
  un paso hacia adelante en dataset de test.

  Argumentos:
    model: Un modelo de PyTorch a ser probado.
    dataloader: Una instancia de DataLoader para probar el modelo.
    loss_fn: Una función de pérdida de PyTorch para calcular la pérdida en los datos de test.
    device: El dispositivo en el que se computará (e.g. "cuda" o "cpu").

  Retorna:
    Una tupla con los valores de pérdida y precisión de la prueba.
    En la forma (test_loss, test_accuracy). Por ejemplo:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 

    # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

    # Turn on inference context manager
  with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    
  """Entrena y prueba un modelo de PyTorch.

  Pasa un modelo objetivo de PyTorch a través de las funciones
  train_step() y test_step() por un número de épocas, entrenando y 
  probando el modelo en el mismo ciclo de época.

  Calcula, imprime y almacena las métricas de evaluación durante el proceso.

  Argumentos:
    model: Un modelo de PyTorch a ser entrenado y probado.
    train_dataloader: Una instancia de DataLoader para entrenar el modelo.
    test_dataloader: Una instancia de DataLoader para probar el modelo.
    optimizer: Un optimizador de PyTorch para ayudar a minimizar la función de pérdida.
    loss_fn: Una función de pérdida de PyTorch para calcular la pérdida en ambos conjuntos de datos.
    epochs: Un entero que indica cuántas épocas entrenar.
    device: El dispositivo en el que se computará (e.g. "cuda" o "cpu").

  Retorna:
    Un diccionario con las pérdidas de entrenamiento y prueba, así como 
    las métricas de precisión de entrenamiento y prueba. Cada métrica 
    tiene un valor en una lista para cada época.
    En la forma: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    Por ejemplo, si entrenamos con epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """ 

  # Diccionarios vacíos
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
    }

  # Loop a través de training y testing steps a partir de epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print de que resultados
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Actualización diccionarios
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Devolver los resultados al final de los epochs
  return results

    