"""
Script para generar la Curva Precision-Recall (Gold Standard)

Evalúa el modelo final en el conjunto de test y genera la curva PR
que muestra el compromiso entre calidad (Precision) y cantidad (Recall).

Uso:
    python scripts/generate_pr_curve.py
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup de paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # IA/
sys.path.insert(0, os.path.join(root_dir, 'src'))

from dataset.dtset import TrafficFlowDataset
from models.architecture import TrafficQuantizerNet
from infer.evaluate_model import (
    decode_predictions, 
    nms, 
    load_ground_truth, 
    calculate_iou,
    plot_precision_recall_curve
)


def calculate_pr_curve(predictions, ground_truths, iou_threshold=0.5):
    """
    Calcula los puntos de la curva Precision-Recall.
    
    Args:
        predictions: Lista de predicciones [(image_id, [boxes])]
        ground_truths: Lista de GT [(image_id, [boxes])]
        iou_threshold: Umbral de IoU para considerar TP
        
    Returns:
        tuple: (precisions, recalls, ap)
    """
    # Aplanar todas las detecciones con sus scores
    all_detections = []
    for img_id, dets in predictions:
        for det in dets:
            all_detections.append((img_id, det))
    
    # Ordenar por score descendente
    all_detections = sorted(all_detections, key=lambda x: x[1][4], reverse=True)
    
    # Diccionario de GT por imagen
    gt_dict = {img_id: boxes for img_id, boxes in ground_truths}
    total_gt = sum(len(boxes) for _, boxes in ground_truths)
    
    # Tracking de GT ya emparejados
    gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in ground_truths}
    
    # Calcular TP y FP para cada threshold de score
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    
    for idx, (img_id, det) in enumerate(all_detections):
        gt_boxes = gt_dict.get(img_id, [])
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[img_id][gt_idx]:
                continue
            iou = calculate_iou(det[:4], gt_box[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[idx] = 1
            gt_matched[img_id][best_gt_idx] = True
        else:
            fp[idx] = 1
    
    # Calcular precision y recall acumulados
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
    
    # Agregar puntos (0, 1) y (1, 0) para completar la curva
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # Hacer la curva monotónica (interpolación)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calcular AP (área bajo la curva)
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return precisions.tolist(), recalls.tolist(), ap


def main():
    """Función principal"""
    print("="*70)
    print("  GENERACIÓN DE CURVA PRECISION-RECALL (GOLD STANDARD)")
    print("="*70)
    
    # Configurar dispositivo
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🖥️  Dispositivo: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n🖥️  Dispositivo: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("\n🖥️  Dispositivo: CPU")
    
    # Paths
    models_dir = os.path.join(root_dir, 'models')
    checkpoint_path = os.path.join(models_dir, 'traffic_model_final.pth')
    
    data_root = os.path.join(root_dir, 'data', 'Intersection-Flow-5K')
    test_img_dir = os.path.join(data_root, 'images', 'test')
    test_label_dir = os.path.join(data_root, 'labels', 'test')
    
    print(f"\n📂 Configuración:")
    print(f"   Modelo: {os.path.basename(checkpoint_path)}")
    print(f"   Test Images: {test_img_dir}")
    print(f"   Test Labels: {test_label_dir}")
    
    # Verificar archivos
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Error: No se encuentra el modelo: {checkpoint_path}")
        return
    
    if not os.path.exists(test_img_dir):
        print(f"\n❌ Error: No se encuentra directorio de test: {test_img_dir}")
        return
    
    # Cargar modelo
    print("\n🔄 Cargando modelo...")
    model = TrafficQuantizerNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Modelo cargado (Época {checkpoint.get('epoch', 'N/A')})")
    
    # Cargar dataset de test
    print("\n📊 Cargando conjunto de test...")
    test_dataset = TrafficFlowDataset(
        img_dir=test_img_dir,
        label_dir=test_label_dir,
        input_size=512,
        stride=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✅ Dataset: {len(test_dataset)} imágenes de test")
    
    # Evaluar modelo
    print("\n🔬 Evaluando modelo en conjunto de test...")
    print("   " + "─"*66)
    
    all_predictions = []
    all_ground_truths = []
    
    score_threshold = 0.3  # Threshold bajo para capturar toda la curva
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Procesando")):
            inputs = batch['input'].to(device)
            
            # Obtener nombre de archivo desde el dataset
            img_file = test_dataset.img_files[idx]
            img_id = os.path.splitext(img_file)[0]
            
            # Predicción
            hm_pred, wh_pred, off_pred = model(inputs)
            
            # Decodificar
            pred_boxes = decode_predictions(
                hm_pred[0].cpu(),
                wh_pred[0].cpu(),
                off_pred[0].cpu(),
                score_threshold=score_threshold,
                max_detections=200
            )
            
            # Aplicar NMS
            pred_boxes = nms(pred_boxes, iou_threshold=0.5)
            
            # Ground truth
            label_path = os.path.join(test_label_dir, img_id + '.txt')
            gt_boxes = load_ground_truth(label_path, output_size=128)
            
            all_predictions.append((img_id, pred_boxes))
            all_ground_truths.append((img_id, gt_boxes))
    
    # Calcular curva PR
    print("\n📈 Calculando Curva Precision-Recall...")
    
    precisions, recalls, ap = calculate_pr_curve(
        all_predictions, 
        all_ground_truths, 
        iou_threshold=0.5
    )
    
    print(f"\n✅ Curva calculada exitosamente")
    print(f"   Average Precision (AP@0.50): {ap:.4f}")
    print(f"   Puntos en la curva: {len(precisions)}")
    
    # Generar gráfico
    print("\n🎨 Generando visualización...")
    
    output_path = os.path.join(root_dir, 'precision_recall_curve.png')
    
    # Preparar datos en formato esperado
    precisions_dict = {0.5: np.array(precisions)}
    recalls_dict = {0.5: np.array(recalls)}
    aps_dict = {0.5: ap}
    
    plot_precision_recall_curve(
        precisions_dict, 
        recalls_dict, 
        aps_dict, 
        save_path=output_path
    )
    
    print("\n" + "="*70)
    print("  ✅ CURVA PRECISION-RECALL GENERADA EXITOSAMENTE")
    print("="*70)
    print(f"\n📊 Gráfico guardado en: {output_path}")
    print("\n💡 Interpretación:")
    print(f"   • AP@0.50 = {ap:.4f} ({'Excelente' if ap > 0.9 else 'Bueno' if ap > 0.7 else 'Aceptable'})")
    print("   • La curva muestra el compromiso entre:")
    print("     - Precision (calidad): De lo que predecimos, ¿cuánto es correcto?")
    print("     - Recall (cantidad): De lo que existe, ¿cuánto encontramos?")
    print("   • Curva cercana a (1.0, 1.0) = Detector robusto")
    print("="*70)


if __name__ == "__main__":
    main()
