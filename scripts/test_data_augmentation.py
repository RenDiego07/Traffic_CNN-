"""
Script para probar y visualizar Data Augmentation con Albumentations

Verifica que las transformaciones de imágenes y bounding boxes sean correctas,
mostrando comparaciones lado a lado de la imagen original vs augmentada.

Uso:
    python scripts/test_data_augmentation.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# Setup de paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # IA/
sys.path.insert(0, os.path.join(root_dir, 'src'))


def load_image_and_labels(img_path, label_path, motorized_ids=[0, 1, 4, 5]):
    """
    Carga imagen y sus bounding boxes en formato YOLO.
    
    Args:
        img_path: Ruta a la imagen
        label_path: Ruta al archivo de labels
        motorized_ids: IDs de clases a considerar
        
    Returns:
        tuple: (imagen_array, lista_de_bboxes, class_labels)
    """
    # Cargar imagen
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Cargar bounding boxes
    bboxes = []
    class_labels = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            
            # Filtrar solo vehículos motorizados
            if class_id in motorized_ids:
                # Coordenadas normalizadas YOLO: center_x, center_y, width, height
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # Convertir de YOLO (center) a Pascal VOC (x_min, y_min, x_max, y_max)
                # y convertir a píxeles absolutos
                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h
                x_max = (cx + bw / 2) * w
                y_max = (cy + bh / 2) * h
                
                # Albumentations espera formato: [x_min, y_min, x_max, y_max] en píxeles
                bboxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(class_id)
    
    return img_array, bboxes, class_labels


def create_augmentation_pipeline(image_size=512):
    """
    Crea pipeline de data augmentation optimizado para detección de objetos.
    Prioriza mantener los bounding boxes válidos.
    
    Args:
        image_size: Tamaño de salida de la imagen
        
    Returns:
        albumentations.Compose: Pipeline de transformaciones
    """
    transform = A.Compose([
        # 1. Transformaciones geométricas SUAVES (para preservar bboxes)
        A.HorizontalFlip(p=0.5),
        
        # Transformación afín más conservadora
        A.ShiftScaleRotate(
            shift_limit=0.05,     # Desplazamiento reducido a 5%
            scale_limit=0.1,      # Escala reducida a ±10%
            rotate_limit=5,       # Rotación muy suave ±5 grados
            border_mode=0,
            p=0.5                 # Probabilidad reducida
        ),
        
        # 2. Transformaciones de color/iluminación (NO afectan bboxes)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
        ], p=0.8),
        
        # 3. Efectos de desenfoque suaves
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        
        # 4. Ruido
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        
        # 5. Simulación de clima (conservadora)
        A.RandomRain(
            blur_value=3,
            brightness_coefficient=0.9,
            p=0.1
        ),
        
        A.RandomFog(
            fog_coef_range=(0.05, 0.15),
            alpha_coef=0.08,
            p=0.05
        ),
        
        # 6. Resize final
        A.Resize(height=image_size, width=image_size),
        
    ], bbox_params=A.BboxParams(
        format='pascal_voc',           
        min_visibility=0.2,            # Reducido a 20% para mantener más bboxes
        min_area=20,                   # Área mínima de 20 píxeles
        label_fields=['class_labels']  
    ))
    
    return transform


def visualize_comparison(original_img, original_bboxes, augmented_img, augmented_bboxes, 
                        original_labels, augmented_labels, save_path=None):
    """
    Visualiza imagen original vs augmentada con bounding boxes.
    
    Args:
        original_img: Imagen original (numpy array)
        original_bboxes: Bboxes originales (formato pascal_voc normalizado)
        augmented_img: Imagen augmentada
        augmented_bboxes: Bboxes augmentadas
        original_labels: Etiquetas de clase originales
        augmented_labels: Etiquetas de clase augmentadas
        save_path: Ruta para guardar la comparación
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Imagen original
    ax1 = axes[0]
    ax1.imshow(original_img)
    ax1.set_title(f'Original ({len(original_bboxes)} vehículos)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Dibujar bboxes originales (ya en píxeles)
    for bbox, label in zip(original_bboxes, original_labels):
        x_min, y_min, x_max, y_max = bbox
        
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Etiqueta de clase
        ax1.text(x_min, y_min - 5, f'ID:{label}', 
                color='lime', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=2))
    
    # Imagen augmentada
    ax2 = axes[1]
    ax2.imshow(augmented_img)
    ax2.set_title(f'Augmentada ({len(augmented_bboxes)} vehículos)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Dibujar bboxes augmentadas (ya en píxeles después de Albumentations)
    for bbox, label in zip(augmented_bboxes, augmented_labels):
        x_min, y_min, x_max, y_max = bbox
        
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax2.add_patch(rect)
        
        # Etiqueta de clase
        ax2.text(x_min, y_min - 5, f'ID:{label}', 
                color='red', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=2))
    
    # Información adicional
    info_text = (
        f"Vehículos originales: {len(original_bboxes)}\n"
        f"Vehículos después de aug: {len(augmented_bboxes)}\n"
        f"Pérdida: {len(original_bboxes) - len(augmented_bboxes)}"
    )
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Comparación guardada: {save_path}")
    
    plt.close(fig)


def main():
    """Función principal"""
    print("="*70)
    print("  TEST DE DATA AUGMENTATION CON ALBUMENTATIONS")
    print("="*70)
    
    # Configurar paths
    data_root = os.path.join(root_dir, 'data', 'Intersection-Flow-5K')
    train_img_dir = os.path.join(data_root, 'images', 'train')
    train_label_dir = os.path.join(data_root, 'labels', 'train')
    output_dir = os.path.join(root_dir, 'src', 'infer', 'results', 'augmentation_tests')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Directorios:")
    print(f"   Train Images: {train_img_dir}")
    print(f"   Train Labels: {train_label_dir}")
    print(f"   Output: {output_dir}")
    
    # Verificar directorios
    if not os.path.exists(train_img_dir):
        print(f"\n❌ Error: No se encuentra {train_img_dir}")
        return
    
    # Obtener lista de imágenes
    img_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not img_files:
        print("\n❌ No se encontraron imágenes")
        return
    
    print(f"\n✅ Encontradas {len(img_files)} imágenes de entrenamiento")
    
    # Crear pipeline de augmentation
    print("\n🔧 Creando pipeline de data augmentation...")
    transform = create_augmentation_pipeline(image_size=512)
    print("✅ Pipeline creado con las siguientes transformaciones:")
    print("   • Flip horizontal (50%)")
    print("   • Shift/Scale/Rotate más conservador (50%)")
    print("   • Brightness/Contrast/HSV/ColorJitter (80%)")
    print("   • Blur suave (20%)")
    print("   • Noise (20%)")
    print("   • Rain (10%) / Fog (5%)")
    print("   • Min visibility: 20% (más permisivo)")
    print("   • Min área: 20px")
    
    # Seleccionar muestras aleatorias
    num_samples = min(10, len(img_files))
    sample_files = random.sample(img_files, num_samples)
    
    print(f"\n🎲 Testeando {num_samples} imágenes aleatorias...")
    print("   " + "─"*66)
    
    successful = 0
    failed = 0
    total_bboxes_original = 0
    total_bboxes_augmented = 0
    
    for i, img_file in enumerate(sample_files, 1):
        img_path = os.path.join(train_img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_label_dir, label_file)
        
        try:
            # Cargar imagen y labels
            img_array, bboxes, class_labels = load_image_and_labels(img_path, label_path)
            
            if len(bboxes) == 0:
                print(f"   ⚠️  [{i}/{num_samples}] {img_file}: Sin vehículos, saltando...")
                continue
            
            # Aplicar augmentation
            transformed = transform(image=img_array, bboxes=bboxes, class_labels=class_labels)
            
            aug_img = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
            
            # Estadísticas
            total_bboxes_original += len(bboxes)
            total_bboxes_augmented += len(aug_bboxes)
            
            # Visualizar comparación
            output_path = os.path.join(output_dir, f'test_{i}_{os.path.splitext(img_file)[0]}.png')
            visualize_comparison(
                img_array, bboxes, aug_img, aug_bboxes,
                class_labels, aug_labels, output_path
            )
            
            print(f"   ✓ [{i}/{num_samples}] {img_file}: {len(bboxes)} → {len(aug_bboxes)} vehículos")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ [{i}/{num_samples}] {img_file}: Error - {e}")
            failed += 1
    
    # Resumen final
    print("\n" + "="*70)
    print("  📊 RESUMEN DE PRUEBAS")
    print("="*70)
    print(f"\n✅ Exitosas: {successful}/{num_samples}")
    print(f"❌ Fallidas: {failed}/{num_samples}")
    print(f"\n📦 Bounding Boxes:")
    print(f"   Original: {total_bboxes_original}")
    print(f"   Después de aug: {total_bboxes_augmented}")
    
    if total_bboxes_original > 0:
        retention = (total_bboxes_augmented / total_bboxes_original) * 100
        print(f"   Retención: {retention:.1f}%")
        
        if retention > 95:
            print("   ✅ Excelente retención de bounding boxes")
        elif retention > 85:
            print("   ✅ Buena retención de bounding boxes")
        else:
            print("   ⚠️  Baja retención, considera ajustar parámetros")
    
    print(f"\n📂 Visualizaciones guardadas en:")
    print(f"   {output_dir}")
    
    print("\n💡 Próximos pasos:")
    print("   1. Revisa las visualizaciones para verificar fidelidad")
    print("   2. Ajusta parámetros del pipeline si es necesario")
    print("   3. Integra el pipeline en tu dataset (dtset.py)")
    print("="*70)


if __name__ == "__main__":
    main()
