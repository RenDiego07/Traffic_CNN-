import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
# Importamos las matemáticas desde utils
from utils.gaussian import draw_gaussian, gaussian_radius

class TrafficFlowDataset(Dataset):
    def __init__(self, img_dir, label_dir, input_size=512, stride=4):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.stride = stride
        self.output_size = input_size // stride # 128
        
        # Filtramos solo archivos de imagen
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # IDs que consideramos "Vehículo Motorizado"
        # Ajusta esto según tu classes.txt (Ej: 0=Vehicle, 1=Bus, 5=Truck)
        self.motorized_ids = [0, 1, 4, 5] 

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. Cargar Imagen
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        
        # Resize básico (512x512)
        img = img.resize((self.input_size, self.input_size))
        
        # Normalizar a [0, 1] y trasponer a (Canales, H, W)
        inp = np.array(img, dtype=np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)

        # 2. Inicializar Targets (Ground Truth)
        # Heatmap: 1 canal (Clase única)
        hm = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        # Size (Width, Height): 2 canales
        wh = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        # Offset (Regresión local x, y): 2 canales
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        # Máscara para saber dónde calcular el error de tamaño/offset
        reg_mask = np.zeros((self.output_size, self.output_size), dtype=np.uint8)

        # 3. Leer Etiquetas
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                class_id = int(parts[0])
                
                # FILTRO: Solo procesamos si es vehículo motorizado
                if class_id in self.motorized_ids:
                    # Coordenadas normalizadas (0-1) del txt
                    # center_x, center_y, w, h
                    norm_cx, norm_cy = float(parts[1]), float(parts[2])
                    norm_w, norm_h = float(parts[3]), float(parts[4])
                    
                    # Convertir al tamaño del grid de salida (128x128)
                    bbox_cx = norm_cx * self.output_size
                    bbox_cy = norm_cy * self.output_size
                    bbox_w = norm_w * self.output_size
                    bbox_h = norm_h * self.output_size
                    
                    # Calcular radio y dibujar gaussiana
                    radius = gaussian_radius((bbox_h, bbox_w))
                    radius = max(0, int(radius))
                    
                    ct_int = np.array([bbox_cx, bbox_cy], dtype=np.int32)
                    
                    # Asegurarnos de que esté dentro de la imagen
                    if (0 <= ct_int[0] < self.output_size) and (0 <= ct_int[1] < self.output_size):
                        draw_gaussian(hm[0], ct_int, radius)
                        
                        # Llenar datos de regresión en el centro
                        wh[0, ct_int[1], ct_int[0]] = bbox_w
                        wh[1, ct_int[1], ct_int[0]] = bbox_h
                        
                        reg[0, ct_int[1], ct_int[0]] = bbox_cx - ct_int[0]
                        reg[1, ct_int[1], ct_int[0]] = bbox_cy - ct_int[1]
                        
                        reg_mask[ct_int[1], ct_int[0]] = 1

        # Retornar todo como tensores de PyTorch
        return {
            'input': torch.from_numpy(inp),
            'hm': torch.from_numpy(hm),
            'wh': torch.from_numpy(wh),
            'reg': torch.from_numpy(reg),
            'reg_mask': torch.from_numpy(reg_mask)
        }