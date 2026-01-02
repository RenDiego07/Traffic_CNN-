"""
Script de Evaluación para TrafficQuantizerNet

Evalúa el modelo entrenado en el conjunto de test y calcula métricas de rendimiento:
    - Precision, Recall, F1-Score
    - Average Precision (AP) por IoU threshold
    - Mean Average Precision (mAP)
    - Visualización de predicciones vs ground truth

Uso:
    # Evaluar modelo final
    python evaluate_model.py
    
    # Evaluar checkpoint específico
    python evaluate_model.py --checkpoint ../models/traffic_model_ep25.pth
    
    # Guardar visualizaciones
    python evaluate_model.py --save-viz --output-dir results/
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import seaborn as sns

# --- SETUP DE IMPORTACIÓN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src/
root_dir = os.path.dirname(parent_dir)     # IA/
sys.path.insert(0, parent_dir)
sys.path.insert(0, root_dir)

from dataset.dtset import TrafficFlowDataset
from models.architecture import TrafficQuantizerNet


def nms(detections, iou_threshold=0.5):
    """
    Non-Maximum Suppression para eliminar detecciones duplicadas.
    
    Args:
        detections: Lista de [x1, y1, x2, y2, score, class]
        iou_threshold: Umbral de IoU para suprimir
        
    Returns:
        Lista de detecciones filtradas
    """
    if len(detections) == 0:
        return []
    
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []
    
    while len(detections) > 0:
        best = detections.pop(0)
        keep.append(best)
        
        # Filtrar detecciones con alto IoU
        filtered = []
        for det in detections:
            iou = calculate_iou(best[:4], det[:4])
            if iou < iou_threshold:
                filtered.append(det)
        detections = filtered
    
    return keep


def calculate_iou(box1, box2):
    """
    Calcula Intersection over Union entre dos boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def decode_predictions(hm, wh, reg, score_threshold=0.15, max_detections=100):
    """
    Decodifica las salidas del modelo en bounding boxes.
    
    Args:
        hm: Heatmap [1, 128, 128]
        wh: Size predictions [2, 128, 128]
        reg: Offset predictions [2, 128, 128]
        score_threshold: Umbral mínimo de confianza
        max_detections: Máximo número de detecciones
        
    Returns:
        Lista de detecciones [x1, y1, x2, y2, score]
    """
    hm = hm.squeeze(0)  # [128, 128]
    batch, height, width = 1, hm.shape[0], hm.shape[1]
    
    # Aplicar max pooling para encontrar picos locales
    kernel = 3
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(
        hm.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=pad
    )
    
    # Mantener solo máximos locales
    keep = (hmax == hm.unsqueeze(0).unsqueeze(0)).float()
    hm = hm * keep.squeeze()
    
    # Obtener top-k detecciones
    scores = hm.view(-1)
    topk_scores, topk_inds = torch.topk(scores, min(max_detections, scores.shape[0]))
    
    # Filtrar por threshold
    mask = topk_scores > score_threshold
    topk_scores = topk_scores[mask]
    topk_inds = topk_inds[mask]
    
    # Convertir índices 1D a coordenadas 2D
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()
    
    # Aplicar offsets
    topk_xs = topk_xs + reg[0, topk_ys.long(), topk_xs.long()]
    topk_ys = topk_ys + reg[1, topk_ys.long(), topk_xs.long()]
    
    # Obtener tamaños
    topk_ws = wh[0, topk_ys.long(), topk_xs.long()]
    topk_hs = wh[1, topk_ys.long(), topk_xs.long()]
    
    # Convertir a formato [x1, y1, x2, y2, score]
    detections = []
    for i in range(len(topk_scores)):
        x_center = topk_xs[i].item()
        y_center = topk_ys[i].item()
        w = topk_ws[i].item()
        h = topk_hs[i].item()
        score = topk_scores[i].item()
        
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        detections.append([x1, y1, x2, y2, score])
    
    return detections


def load_ground_truth(label_path, output_size=128, motorized_ids=[0, 1, 4, 5]):
    """
    Carga ground truth desde archivo YOLO format.
    
    Args:
        label_path: Ruta al archivo .txt
        output_size: Tamaño del grid (128)
        motorized_ids: IDs de clases motorizadas
        
    Returns:
        Lista de boxes [x1, y1, x2, y2]
    """
    boxes = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            if class_id in motorized_ids:
                norm_cx, norm_cy = float(parts[1]), float(parts[2])
                norm_w, norm_h = float(parts[3]), float(parts[4])
                
                # Convertir a coordenadas del grid
                cx = norm_cx * output_size
                cy = norm_cy * output_size
                w = norm_w * output_size
                h = norm_h * output_size
                
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                boxes.append([x1, y1, x2, y2])
    
    return boxes


def calculate_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Calcula Average Precision para un umbral de IoU dado.
    
    Args:
        predictions: Lista de todas las predicciones [(image_id, [boxes])]
        ground_truths: Lista de todos los GT [(image_id, [boxes])]
        iou_threshold: Umbral de IoU para considerar TP
        
    Returns:
        float: Average Precision
    """
    # Ordenar predicciones por score
    all_detections = []
    for img_id, dets in predictions:
        for det in dets:
            all_detections.append((img_id, det))
    
    all_detections = sorted(all_detections, key=lambda x: x[1][4], reverse=True)
    
    # Contadores
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    
    # Diccionario de GT por imagen
    gt_dict = {img_id: boxes for img_id, boxes in ground_truths}
    gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in ground_truths}
    
    # Evaluar cada detección
    for idx, (img_id, det) in enumerate(all_detections):
        gt_boxes = gt_dict.get(img_id, [])
        matched = gt_matched.get(img_id, [])
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if matched[gt_idx]:
                continue
            
            iou = calculate_iou(det[:4], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if not matched[best_gt_idx]:
                tp[idx] = 1
                matched[best_gt_idx] = True
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1
    
    # Calcular precision y recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    total_gt = sum(len(boxes) for _, boxes in ground_truths)
    
    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
    
    # Calcular AP (área bajo la curva precision-recall)
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Hacer la curva monotónica
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calcular área
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap, precisions, recalls


def visualize_predictions(image, pred_boxes, gt_boxes, score_threshold=0.15):
    """
    Visualiza predicciones y ground truth en la imagen.
    
    Args:
        image: PIL Image (512x512)
        pred_boxes: Lista de predicciones [x1, y1, x2, y2, score]
        gt_boxes: Lista de GT [x1, y1, x2, y2]
        score_threshold: Umbral para visualizar
        
    Returns:
        PIL Image con visualización
    """
    # Escalar del grid (128) a imagen (512)
    scale = 512 / 128
    
    draw = ImageDraw.Draw(image)
    
    # Dibujar GT en verde
    for box in gt_boxes:
        x1, y1, x2, y2 = [coord * scale for coord in box]
        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
    
    # Dibujar predicciones en rojo
    for box in pred_boxes:
        if box[4] >= score_threshold:
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = [coord * scale for coord in [x1, y1, x2, y2]]
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            # Agregar score
            draw.text((x1, y1 - 10), f'{score:.2f}', fill='red')
    
    return image


def calculate_confusion_matrix(predictions, ground_truths, iou_threshold=0.5):
    """
    Calcula la matriz de confusión para detección de objetos.
    
    Args:
        predictions: Lista de predicciones [(image_id, [boxes])]
        ground_truths: Lista de GT [(image_id, [boxes])]
        iou_threshold: Umbral de IoU para considerar TP
        
    Returns:
        dict: TP, FP, FN, TN (TN=0 para detección), precision, recall, f1
    """
    tp = 0  # True Positives: detecciones correctas
    fp = 0  # False Positives: detecciones incorrectas
    fn = 0  # False Negatives: objetos no detectados
    
    # Crear diccionario de GT por imagen
    gt_dict = {img_id: boxes for img_id, boxes in ground_truths}
    pred_dict = {img_id: boxes for img_id, boxes in predictions}
    
    # Procesar cada imagen
    for img_id, gt_boxes in ground_truths:
        pred_boxes = pred_dict.get(img_id, [])
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)
        
        # Evaluar cada predicción contra cada GT
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Si encontró un match válido
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
            else:
                fp += 1  # Detección sin GT correspondiente
        
        # Contar GT no detectados
        fn += sum(1 for matched in gt_matched if not matched)
    
    # Calcular métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': 0,  # No aplica para detección de objetos
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def plot_confusion_matrix(confusion_metrics, save_path=None):
    """
    Visualiza la matriz de confusión.
    
    Args:
        confusion_metrics: Dict con TP, FP, FN, TN
        save_path: Ruta donde guardar la imagen (opcional)
    """
    # Crear matriz 2x2 (para visualización, aunque TN=0)
    # Filas: Actual (GT), Columnas: Predicho
    cm = np.array([
        [confusion_metrics['TP'], confusion_metrics['FN']],  # Actual: Positivo
        [confusion_metrics['FP'], confusion_metrics['TN']]   # Actual: Negativo
    ])
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Detected', 'Not Detected'],
                yticklabels=['Vehicle (GT)', 'Background'],
                cbar_kws={'label': 'Count'},
                ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
    
    # Títulos y etiquetas
    ax.set_title('Confusion Matrix - Vehicle Detection', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual (Ground Truth)', fontsize=14, fontweight='bold')
    
    # Agregar métricas en el título
    metrics_text = f"Precision: {confusion_metrics['precision']:.3f}  |  "
    metrics_text += f"Recall: {confusion_metrics['recall']:.3f}  |  "
    metrics_text += f"F1-Score: {confusion_metrics['f1_score']:.3f}"
    
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Matriz de confusión guardada: {save_path}")
    
    plt.close(fig)


def calculate_counting_metrics(predictions, ground_truths):
    """
    Calcula métricas de conteo de vehículos.
    
    Args:
        predictions: Lista de predicciones [(image_id, [boxes])]
        ground_truths: Lista de GT [(image_id, [boxes])]
        
    Returns:
        dict: MAE, RMSE, y otros estadísticos
    """
    pred_counts = np.array([len(boxes) for _, boxes in predictions])
    gt_counts = np.array([len(boxes) for _, boxes in ground_truths])
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_counts - gt_counts))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))
    
    # Estadísticas adicionales
    errors = pred_counts - gt_counts
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Porcentaje de imágenes con conteo exacto
    exact_match = np.sum(pred_counts == gt_counts) / len(pred_counts) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'mean_error': mean_error,
        'std_error': std_error,
        'exact_match_percentage': exact_match,
        'pred_counts': pred_counts,
        'gt_counts': gt_counts
    }


def visualize_heatmaps(image, pred_hm, gt_hm, pred_boxes, gt_boxes, score_threshold=0.15):
    """
    Crea una visualización comparativa de heatmaps predichos vs ground truth.
    
    Args:
        image: PIL Image (512x512)
        pred_hm: Heatmap predicho [1, 128, 128]
        gt_hm: Heatmap ground truth [1, 128, 128]
        pred_boxes: Lista de predicciones [x1, y1, x2, y2, score]
        gt_boxes: Lista de GT [x1, y1, x2, y2]
        score_threshold: Umbral para visualizar boxes
        
    Returns:
        matplotlib Figure con 4 subplots
    """
    # Convertir imagen PIL a numpy
    img_np = np.array(image)
    
    # Extraer heatmaps y convertir a numpy
    pred_hm_np = pred_hm.squeeze(0).cpu().numpy() if torch.is_tensor(pred_hm) else pred_hm.squeeze(0)
    gt_hm_np = gt_hm.squeeze(0).cpu().numpy() if torch.is_tensor(gt_hm) else gt_hm.squeeze(0)
    
    # Escalar heatmaps a tamaño de imagen (128 -> 512)
    pred_hm_resized = cv2.resize(pred_hm_np, (512, 512))
    gt_hm_resized = cv2.resize(gt_hm_np, (512, 512))
    
    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Imagen original con GT boxes
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Ground Truth Boxes', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    scale = 512 / 128
    for box in gt_boxes:
        x1, y1, x2, y2 = [coord * scale for coord in box]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=2)
        axes[0, 0].add_patch(rect)
    
    # 2. Imagen original con predicciones
    axes[0, 1].imshow(img_np)
    axes[0, 1].set_title(f'Predicted Boxes (threshold={score_threshold})', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    for box in pred_boxes:
        if box[4] >= score_threshold:
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = [coord * scale for coord in [x1, y1, x2, y2]]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
            axes[0, 1].add_patch(rect)
            axes[0, 1].text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, fontweight='bold')
    
    # 3. Ground Truth Heatmap
    axes[1, 0].imshow(img_np, alpha=0.5)
    hm_gt = axes[1, 0].imshow(gt_hm_resized, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(hm_gt, ax=axes[1, 0], fraction=0.046)
    
    # 4. Predicted Heatmap
    axes[1, 1].imshow(img_np, alpha=0.5)
    hm_pred = axes[1, 1].imshow(pred_hm_resized, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[1, 1].set_title('Predicted Heatmap', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(hm_pred, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    
    return fig


def save_heatmap_comparison(image, pred_hm, gt_hm, pred_boxes, gt_boxes, save_path, score_threshold=0.15):
    """
    Guarda la visualización comparativa de heatmaps.
    
    Args:
        image: PIL Image
        pred_hm: Heatmap predicho
        gt_hm: Heatmap ground truth
        pred_boxes: Predicciones
        gt_boxes: Ground truth boxes
        save_path: Ruta donde guardar
        score_threshold: Umbral de visualización
    """
    fig = visualize_heatmaps(image, pred_hm, gt_hm, pred_boxes, gt_boxes, score_threshold)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def evaluate(checkpoint_path, save_viz=False, save_heatmaps=False, output_dir='results/', score_threshold=0.15, nms_threshold=0.5):
    """
    Función principal de evaluación.
    
    Evalúa el modelo en métricas de:
        - Detección: IoU, mAP, Precision, Recall
        - Conteo: MAE, RMSE
        - Visualización: Heatmaps predichos vs ground truth
    """
    
    # ========================================================================
    # 1. CONFIGURACIÓN
    # ========================================================================
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Evaluando en Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Evaluando en NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️  Evaluando en CPU")
    
    # ========================================================================
    # 2. CARGAR MODELO
    # ========================================================================
    print(f"\n📂 Cargando modelo: {checkpoint_path}")
    
    model = TrafficQuantizerNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modelo cargado (Época {checkpoint.get('epoch', 'N/A')})")
    
    # ========================================================================
    # 3. CARGAR DATASET DE TEST
    # ========================================================================
    data_root = os.path.join(root_dir, 'data', 'Intersection-Flow-5K')
    test_img_dir = os.path.join(data_root, 'images', 'test')
    test_label_dir = os.path.join(data_root, 'labels', 'test')
    
    print(f"\n📂 Cargando dataset de test:")
    print(f"   - Imágenes: {test_img_dir}")
    print(f"   - Labels: {test_label_dir}")
    
    test_dataset = TrafficFlowDataset(
        img_dir=test_img_dir,
        label_dir=test_label_dir,
        input_size=512,
        stride=4
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"✅ Dataset cargado: {len(test_dataset)} imágenes")
    
    # ========================================================================
    # 4. EVALUACIÓN
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"INICIANDO EVALUACIÓN")
    print(f"{'='*60}\n")
    
    all_predictions = []
    all_ground_truths = []
    
    if save_viz or save_heatmaps:
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 Guardando visualizaciones en: {output_dir}")
        if save_heatmaps:
            heatmap_dir = os.path.join(output_dir, 'heatmaps')
            os.makedirs(heatmap_dir, exist_ok=True)
            print(f"🔥 Guardando comparaciones de heatmaps en: {heatmap_dir}")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Evaluando")):
            # Forward pass
            inputs = batch['input'].to(device)
            hm_pred, wh_pred, reg_pred = model(inputs)
            
            # Decodificar predicciones
            pred_boxes = decode_predictions(
                hm_pred[0].cpu(),
                wh_pred[0].cpu(),
                reg_pred[0].cpu(),
                score_threshold=score_threshold
            )
            
            # Aplicar NMS
            pred_boxes = nms(pred_boxes, iou_threshold=nms_threshold)
            
            # Cargar ground truth
            img_file = test_dataset.img_files[idx]
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(test_label_dir, label_file)
            gt_boxes = load_ground_truth(label_path)
            
            # Obtener ground truth heatmap del batch
            gt_hm = batch['hm'][0]  # [1, 128, 128]
            
            # Guardar para cálculo de métricas
            all_predictions.append((idx, pred_boxes))
            all_ground_truths.append((idx, gt_boxes))
            
            # Cargar imagen original
            img_path = os.path.join(test_img_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512))
            
            # Visualizar boxes (primeras 20 imágenes)
            if save_viz and idx < 20:
                vis_img = visualize_predictions(img.copy(), pred_boxes, gt_boxes, score_threshold)
                vis_path = os.path.join(output_dir, f"pred_{idx:04d}.jpg")
                vis_img.save(vis_path)
            
            # Visualizar heatmaps (primeras 20 imágenes)
            if save_heatmaps and idx < 20:
                heatmap_path = os.path.join(heatmap_dir, f"heatmap_{idx:04d}.png")
                save_heatmap_comparison(
                    img.copy(), 
                    hm_pred[0].cpu(), 
                    gt_hm, 
                    pred_boxes, 
                    gt_boxes, 
                    heatmap_path,
                    score_threshold
                )
    
    # ========================================================================
    # 5. CALCULAR MÉTRICAS
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"CALCULANDO MÉTRICAS")
    print(f"{'='*60}\n")
    
    # ========================================================================
    # 5.1 MÉTRICAS DE DETECCIÓN (VISIÓN)
    # ========================================================================
    print("🔍 MÉTRICAS DE DETECCIÓN (Visión por Computadora)")
    print("─" * 60)
    
    # Calcular AP para diferentes thresholds de IoU
    iou_thresholds = [0.5, 0.75, 0.9]
    aps = {}
    
    for iou_thresh in iou_thresholds:
        ap, precisions, recalls = calculate_ap(
            all_predictions, 
            all_ground_truths, 
            iou_threshold=iou_thresh
        )
        aps[iou_thresh] = ap
        print(f"   AP @ IoU={iou_thresh:.2f}: {ap:.4f}")
    
    # Calcular mAP
    map_score = np.mean(list(aps.values()))
    print(f"\n   📊 Mean Average Precision (mAP): {map_score:.4f}")
    
    # ========================================================================
    # 5.2 MÉTRICAS DE CONTEO
    # ========================================================================
    print(f"\n🔢 MÉTRICAS DE CONTEO")
    print("─" * 60)
    
    counting_metrics = calculate_counting_metrics(all_predictions, all_ground_truths)
    
    print(f"   Mean Absolute Error (MAE): {counting_metrics['MAE']:.4f}")
    print(f"   Root Mean Square Error (RMSE): {counting_metrics['RMSE']:.4f}")
    print(f"   Mean Error (bias): {counting_metrics['mean_error']:+.4f}")
    print(f"   Std Error: {counting_metrics['std_error']:.4f}")
    print(f"   Exact Match: {counting_metrics['exact_match_percentage']:.2f}%")
    
    # ========================================================================
    # 5.3 MATRIZ DE CONFUSIÓN
    # ========================================================================
    print(f"\n🎯 MATRIZ DE CONFUSIÓN (IoU=0.5)")
    print("─" * 60)
    
    confusion_metrics = calculate_confusion_matrix(
        all_predictions, 
        all_ground_truths, 
        iou_threshold=0.5
    )
    
    print(f"   True Positives (TP):  {confusion_metrics['TP']:>6d}")
    print(f"   False Positives (FP): {confusion_metrics['FP']:>6d}")
    print(f"   False Negatives (FN): {confusion_metrics['FN']:>6d}")
    print(f"")
    print(f"   Precision: {confusion_metrics['precision']:.4f}")
    print(f"   Recall:    {confusion_metrics['recall']:.4f}")
    print(f"   F1-Score:  {confusion_metrics['f1_score']:.4f}")
    
    # Guardar visualización de matriz de confusión
    if save_viz or save_heatmaps:
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(confusion_metrics, save_path=cm_path)
    
    # ========================================================================
    # 5.4 ESTADÍSTICAS GENERALES
    # ========================================================================
    print(f"\n📈 ESTADÍSTICAS GENERALES")
    print("─" * 60)
    
    total_predictions = sum(len(dets) for _, dets in all_predictions)
    total_gt = sum(len(boxes) for _, boxes in all_ground_truths)
    avg_predictions = total_predictions / len(all_predictions)
    avg_gt = total_gt / len(all_ground_truths)
    
    print(f"   Total predicciones: {total_predictions}")
    print(f"   Total ground truth: {total_gt}")
    print(f"   Promedio predicciones/imagen: {avg_predictions:.2f}")
    print(f"   Promedio GT/imagen: {avg_gt:.2f}")
    
    # ========================================================================
    # 6. RESULTADOS
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"✅ EVALUACIÓN COMPLETADA")
    print(f"{'='*60}\n")
    
    return {
        # Métricas de detección
        'mAP': map_score,
        'AP@0.5': aps[0.5],
        'AP@0.75': aps[0.75],
        'AP@0.9': aps[0.9],
        # Matriz de confusión
        'TP': confusion_metrics['TP'],
        'FP': confusion_metrics['FP'],
        'FN': confusion_metrics['FN'],
        'precision': confusion_metrics['precision'],
        'recall': confusion_metrics['recall'],
        'f1_score': confusion_metrics['f1_score'],
        # Métricas de conteo
        'MAE': counting_metrics['MAE'],
        'RMSE': counting_metrics['RMSE'],
        'mean_error': counting_metrics['mean_error'],
        'exact_match_percentage': counting_metrics['exact_match_percentage'],
        # Estadísticas
        'total_predictions': total_predictions,
        'total_gt': total_gt
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluar TrafficQuantizerNet en conjunto de test')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Ruta al checkpoint (por defecto: models/traffic_model_final.pth)'
    )
    parser.add_argument(
        '--save-viz',
        action='store_true',
        help='Guardar visualizaciones de predicciones'
    )
    parser.add_argument(
        '--save-heatmaps',
        action='store_true',
        help='Guardar comparaciones de heatmaps (predichos vs ground truth)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/',
        help='Directorio para guardar resultados'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.15,
        help='Umbral mínimo de confianza (default: 0.15 para capturar más detecciones)'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.5,
        help='Umbral de IoU para NMS'
    )
    
    args = parser.parse_args()
    
    # Si no se especifica checkpoint, usar el modelo final
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models',
            'traffic_model_final.pth'
        )
    
    # Ejecutar evaluación
    results = evaluate(
        checkpoint_path=args.checkpoint,
        save_viz=args.save_viz,
        save_heatmaps=args.save_heatmaps,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold
    )
