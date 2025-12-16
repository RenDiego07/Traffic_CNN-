import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Agregar el directorio raíz del proyecto al path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset.dtset import TrafficDataset
from src.models.cnn_simple import yolo_to_grid_targets


def visualize_grid_targets(idx=0):
    """
    Visualiza cómo se convierten los labels YOLO a grid targets.
    """
    # Cargar dataset
    dataset = TrafficDataset(
        images_dir="../data/Intersection-Flow-5K/proccesed/train/",
        labels_dir="../data/Intersection-Flow-5K/labels/train"
    )
    
    # Obtener una muestra
    img_tensor, boxes, classes, fname = dataset[idx]
    
    # Convertir a grid targets
    obj_target, bbox_target, class_target = yolo_to_grid_targets(boxes, classes, grid_size=16)
    
    # Convertir imagen tensor a numpy
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Imagen original con bounding boxes YOLO
    axes[0].imshow(img_np)
    axes[0].set_title(f"Imagen Original\n{fname}")
    img_h, img_w = img_np.shape[:2]
    
    for i in range(len(boxes)):
        x_c, y_c, w, h = boxes[i]
        cls = classes[i].item()
        
        # Convertir a píxeles
        x_c_px = x_c * img_w
        y_c_px = y_c * img_h
        w_px = w * img_w
        h_px = h * img_h
        
        x1 = x_c_px - w_px / 2
        y1 = y_c_px - h_px / 2
        
        rect = patches.Rectangle(
            (x1, y1), w_px, h_px,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[0].add_patch(rect)
        axes[0].plot(x_c_px, y_c_px, 'ro', markersize=5)
        axes[0].text(x1, y1-5, f"Class {cls}", color='red', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7))
    
    axes[0].axis('off')
    
    # 2. Heatmap de objectness
    axes[1].imshow(img_np)
    im = axes[1].imshow(obj_target.numpy(), alpha=0.6, cmap='hot', interpolation='nearest')
    axes[1].set_title("Grid Objectness Target\n(1 = objeto presente)")
    axes[1].grid(True, which='both', color='cyan', linewidth=0.5, alpha=0.3)
    axes[1].set_xticks(np.arange(-0.5, 16, 1))
    axes[1].set_yticks(np.arange(-0.5, 16, 1))
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # 3. Grid con información detallada
    axes[2].imshow(img_np)
    axes[2].set_title("Grid Cells (16x16)\ncon bounding box offsets")
    
    cell_w = img_w / 16
    cell_h = img_h / 16
    
    # Dibujar grid
    for i in range(17):
        axes[2].axhline(i * cell_h, color='cyan', linewidth=0.5, alpha=0.5)
        axes[2].axvline(i * cell_w, color='cyan', linewidth=0.5, alpha=0.5)
    
    # Marcar celdas con objetos
    for i in range(16):
        for j in range(16):
            if obj_target[i, j] == 1:
                # Dibujar celda
                rect = patches.Rectangle(
                    (j * cell_w, i * cell_h), cell_w, cell_h,
                    linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.3
                )
                axes[2].add_patch(rect)
                
                # Obtener bbox target
                x_offset = bbox_target[i, j, 0].item()
                y_offset = bbox_target[i, j, 1].item()
                width = bbox_target[i, j, 2].item()
                height = bbox_target[i, j, 3].item()
                
                # Calcular centro del objeto en píxeles
                x_center = (j + x_offset) / 16 * img_w
                y_center = (i + y_offset) / 16 * img_h
                
                # Dibujar centro
                axes[2].plot(x_center, y_center, 'go', markersize=8)
                
                # Añadir texto con información
                info_text = f"({i},{j})\nΔx:{x_offset:.2f}\nΔy:{y_offset:.2f}\nw:{width:.2f}\nh:{height:.2f}"
                axes[2].text(j * cell_w + 5, i * cell_h + 10, info_text,
                            color='white', fontsize=6,
                            bbox=dict(facecolor='black', alpha=0.7))
    
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("../../outputs/grid_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Imprimir información
    print(f"\nImagen: {fname}")
    print(f"Dimensiones: {img_w}x{img_h}")
    print(f"Número de objetos: {len(boxes)}")
    print(f"\nDetalles de cada objeto:")
    for i in range(len(boxes)):
        x_c, y_c, w, h = boxes[i]
        cls = classes[i].item()
        
        cell_x = int(x_c * 16)
        cell_y = int(y_c * 16)
        
        print(f"\nObjeto {i+1}:")
        print(f"  Clase: {cls}")
        print(f"  YOLO coords: x={x_c:.3f}, y={y_c:.3f}, w={w:.3f}, h={h:.3f}")
        print(f"  Celda asignada: ({cell_y}, {cell_x})")
        print(f"  Offsets: x={bbox_target[cell_y, cell_x, 0]:.3f}, y={bbox_target[cell_y, cell_x, 1]:.3f}")


if __name__ == "__main__":
    # Crear directorio de outputs
    os.makedirs("../../outputs", exist_ok=True)
    
    # Visualizar varios ejemplos
    print("Visualizando ejemplo 0...")
    visualize_grid_targets(idx=0)
    
    print("\n" + "="*60)
    print("\nVisualizando ejemplo 5...")
    visualize_grid_targets(idx=5)
