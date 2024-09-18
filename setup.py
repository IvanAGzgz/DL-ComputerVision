import os
import subprocess
import zipfile
import shutil
import random
from pathlib import Path

# Configurar la ruta para la carpeta de datos
ruta_datos = Path("data/")
ruta_cancer_data = ruta_datos / "cancer_data"

# Si la carpeta cancer_data no existe, descarga el dataset y prepáralo...
if ruta_cancer_data.is_dir():
    print(f"El directorio {ruta_cancer_data} ya existe.")
else:
    print(f"No se encontró el directorio {ruta_cancer_data}, creándolo...")
    ruta_cancer_data.mkdir(parents=True, exist_ok=True)

# Descargar el dataset de detección de tumores cerebrales desde Kaggle
ruta_archivo_zip = ruta_datos / "brain-mri-images-for-brain-tumor-detection.zip"
if not ruta_archivo_zip.exists():
    print("Descargando el dataset de imágenes MRI de tumores cerebrales desde Kaggle...")
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'navoneel/brain-mri-images-for-brain-tumor-detection', '-p', str(ruta_datos)], check=True)

# Descomprimir el dataset de tumores cerebrales
with zipfile.ZipFile(ruta_archivo_zip, "r") as zip_ref:
    print(f"Descomprimiendo el dataset en {ruta_cancer_data}...")
    zip_ref.extractall(ruta_cancer_data)

# Eliminar el archivo zip
os.remove(ruta_archivo_zip)
print(f"Archivo zip {ruta_archivo_zip} eliminado.")

# Configurar rutas
ruta_datos = Path("data/")
ruta_cancer_data = ruta_datos / "cancer_data"
ruta_sin_cancer = ruta_cancer_data / "no"
ruta_con_cancer = ruta_cancer_data / "yes"

# Crear directorios de destino para entrenamiento y prueba
for category in ["train", "test"]:
    for label in ["sin_cancer", "con_cancer"]:
        path = ruta_cancer_data / category / label
        path.mkdir(parents=True, exist_ok=True)

# Función para mover imágenes con una proporción fija
def mover_imagenes(ruta_origen, ruta_destino_train, ruta_destino_test, proporcion_test=0.2):
    # Listar todas las imágenes en el directorio de origen
    imagenes = list(ruta_origen.glob("*"))
    
    # Mezclar imágenes y calcular el número de imágenes para prueba
    random.shuffle(imagenes)
    num_test = int(len(imagenes) * proporcion_test)
    
    # Dividir imágenes en entrenamiento y prueba
    imagenes_test = imagenes[:num_test]
    imagenes_train = imagenes[num_test:]
    
    # Mover imágenes a la carpeta de entrenamiento
    for img in imagenes_train:
        shutil.move(str(img), ruta_destino_train / img.name)
    
    # Mover imágenes a la carpeta de prueba
    for img in imagenes_test:
        shutil.move(str(img), ruta_destino_test / img.name)

# Dividir y mover las imágenes sin cáncer
mover_imagenes(
    ruta_sin_cancer,
    ruta_cancer_data / "train" / "sin_cancer",
    ruta_cancer_data / "test" / "sin_cancer"
)

# Dividir y mover las imágenes con cáncer
mover_imagenes(
    ruta_con_cancer,
    ruta_cancer_data / "train" / "con_cancer",
    ruta_cancer_data / "test" / "con_cancer"
)

print("Imágenes divididas y movidas exitosamente.")