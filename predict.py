import torch
import torchvision
import argparse

import model_builder

# Creando un parser
parser = argparse.ArgumentParser()

# Obtener la ruta de la imagen
parser.add_argument("--image",
                    help="target image filepath to predict on")

# Obtener la ruta del modelo
parser.add_argument("--model_path",
                    default="models/model.pth",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

# Setup class names
class_names = ["no", "yes"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Obtener la ruta de la imagen
IMG_PATH = args.image
print(f"[INFO] Predicting on {IMG_PATH}")

# Función para cargar el modelo
def load_model(filepath=args.model_path):
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=128,
                                output_shape=2).to(device)

  print(f"[INFO] Cargando modelo: {filepath}")                               
  model.load_state_dict(torch.load(filepath,weights_only=True))

  return model

# Función para cargar el modelo y predecir en una imagen seleccionada
def predict_on_image(image_path=IMG_PATH, filepath=args.model_path):
  # Cargar modelo
  model = load_model(filepath)

  # Cargar la imagen y convertirla a torch.float32 (mismo tipo que el modelo)
  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

  # Preprocesar la imagen para conseguir entre 0 y 1
  image = image / 255.

  # Resize en la imagen para estar al mismo size que el modelo
  transform = torchvision.transforms.Resize(size=(64, 64))
  image = transform(image) 

  # Predecir la imagen
  model.eval()
  with torch.inference_mode():
    # Situarla en el target device
    image = image.to(device)

    # Obtener logits
    pred_logits = model(image.unsqueeze(dim=0)) 

    # Obtener pred probs
    pred_prob = torch.softmax(pred_logits, dim=1)

    # Obtener pred labels
    pred_label = torch.argmax(pred_prob, dim=1)
    pred_label_class = class_names[pred_label]

  print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
  predict_on_image()
