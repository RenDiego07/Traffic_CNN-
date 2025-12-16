import os
from PIL import Image
import shutil

# Rutas de entrada
IMAGES_DIR = "/Users/dfflores/Developer/IA/data/Intersection-Flow-5K/images/train"

# Rutas de salida
OUT_IMAGES_DIR = "/Users/dfflores/Developer/IA/data/Intersection-Flow-5K/images/512pixel/train"
OUT_LABELS_DIR = "/Users/dfflores/Developer/IA/data/Intersection-Flow-5K/labels/512pixel/train"
IMG_SIZE = 512  # tamaño deseado

os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# Listar solo archivos de imagen
valid_exts = (".jpg", ".jpeg", ".png")

image_files = [
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(valid_exts)
]

print(f"Total imágenes encontradas: {len(image_files)}")

for fname in image_files:
    img_path = os.path.join(IMAGES_DIR, fname)
    out_img_path = os.path.join(OUT_IMAGES_DIR, fname)

    # 1. Cargar imagen
    img = Image.open(img_path).convert("RGB")

    # 2. Redimensionar a 512x512 (sin mantener aspecto, tal como decidimos)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    # 3. Guardar imagen redimensionada
    img_resized.save(out_img_path, format="JPEG", quality=95)
