from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image

# --- IMPORTS PARA INTELIGENCIA ARTIFICIAL ---
import torch
import torch.nn.functional as F
from src.models.architecture import TrafficQuantizerNet


# --------------------------------------------------------------------------
# CARGA DEL MODELO
# (Se ejecuta solo una vez cuando la aplicación inicia)
# --------------------------------------------------------------------------
print("Cargando el modelo de IA...")
model_path = 'src/saved_models/traffic_model.pth'
model = TrafficQuantizerNet()
try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Modelo PyTorch cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo PyTorch: {e}")
    model = None


# --------------------------------------------------------------------------
# FUNCIONES DE PROCESAMIENTO
# --------------------------------------------------------------------------
def preprocess_image(img: Image.Image):
    """
    Preprocesa la imagen para que coincida con la entrada del modelo (512x512).
    
    El modelo espera:
    - Tamaño: 512x512
    - Formato: Tensor (C, H, W)
    - Normalización: [0, 1] (no ImageNet, el modelo ya maneja esto)
    """
    # Redimensionar a 512x512
    img_resized = img.resize((512, 512))
    
    # Convertir a array numpy y normalizar a [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Transponer a (C, H, W) y agregar batch dimension
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    
    return img_tensor


def decode_predictions(hm, wh, reg, score_threshold=0.15, max_detections=100):
    """
    Decodifica las salidas del modelo (heatmap, sizes, offsets) a bounding boxes.
    
    Basado en CenterNet: "Objects as Points" (Zhou et al., 2019)
    
    Args:
        hm: Heatmap [128, 128] - probabilidad de centro de objeto
        wh: Size predictions [2, 128, 128] - (width, height)
        reg: Offset predictions [2, 128, 128] - corrección subpíxel (dx, dy)
        score_threshold: Umbral mínimo de confianza
        max_detections: Máximo número de detecciones
        
    Returns:
        Lista de detecciones [x1, y1, x2, y2, score]
    """
    hm = hm.squeeze(0)  # [128, 128]
    height, width = hm.shape
    
    # Aplicar max pooling para encontrar picos locales (Non-Maximum Suppression)
    kernel = 3
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(
        hm.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=pad
    )
    
    # Mantener solo máximos locales
    keep = (hmax == hm.unsqueeze(0).unsqueeze(0)).float()
    hm = hm * keep.squeeze()
    
    # Obtener top-k detecciones por score
    scores = hm.view(-1)
    topk_scores, topk_inds = torch.topk(scores, min(max_detections, scores.shape[0]))
    
    # Filtrar por threshold
    mask = topk_scores > score_threshold
    topk_scores = topk_scores[mask]
    topk_inds = topk_inds[mask]
    
    if len(topk_scores) == 0:
        return []
    
    # Convertir índices 1D a coordenadas 2D en el grid (128x128)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()
    
    # Aplicar offsets para refinar la posición
    topk_xs = topk_xs + reg[0, topk_ys.long(), topk_xs.long()]
    topk_ys = topk_ys + reg[1, topk_ys.long(), topk_xs.long()]
    
    # Obtener tamaños (width, height)
    topk_ws = wh[0, topk_ys.long(), topk_xs.long()]
    topk_hs = wh[1, topk_ys.long(), topk_xs.long()]
    
    # Convertir de coordenadas de centro a formato [x1, y1, x2, y2, score]
    # Nota: Coordenadas están en el espacio del grid 128x128
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


def scale_boxes_to_image(boxes, original_size):
    """
    Escala las coordenadas de bounding boxes del grid (128x128) 
    al tamaño original de la imagen.
    
    Args:
        boxes: Lista de [x1, y1, x2, y2, score] en coordenadas de grid (128x128)
        original_size: Tupla (width, height) de la imagen original
        
    Returns:
        Lista de boxes escalados al tamaño original
    """
    # El modelo trabaja con un grid de 128x128
    # Las imágenes de entrada son 512x512
    # Factor de escala: 512 / 128 = 4
    grid_size = 128
    processed_size = 512
    scale = processed_size / grid_size
    
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2, score = box
        
        # Escalar coordenadas
        x1_scaled = (x1 * scale) * (original_size[0] / processed_size)
        y1_scaled = (y1 * scale) * (original_size[1] / processed_size)
        x2_scaled = (x2 * scale) * (original_size[0] / processed_size)
        y2_scaled = (y2 * scale) * (original_size[1] / processed_size)
        
        # Asegurar que están dentro de los límites
        x1_scaled = max(0, min(x1_scaled, original_size[0]))
        y1_scaled = max(0, min(y1_scaled, original_size[1]))
        x2_scaled = max(0, min(x2_scaled, original_size[0]))
        y2_scaled = max(0, min(y2_scaled, original_size[1]))
        
        scaled_boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, score])
    
    return scaled_boxes


# --------------------------------------------------------------------------
# APLICACIÓN FLASK
# --------------------------------------------------------------------------
app = Flask(__name__)

# Directorio para guardar las imágenes que se suban
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe una imagen, la procesa con el modelo entrenado 
    y devuelve el conteo de vehículos y sus bounding boxes.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    if file and model is not None:
        try:
            # Abrir la imagen y convertir a RGB
            img = Image.open(file.stream).convert("RGB")
            original_width, original_height = img.size
            
            # --- LÓGICA DE INFERENCIA DEL MODELO ---
            
            # 1. Pre-procesar la imagen (resize a 512x512, normalizar a [0,1])
            preprocessed_img = preprocess_image(img)

            # 2. Realizar la predicción
            with torch.no_grad():
                hm_output, wh_output, reg_output = model(preprocessed_img)
            
            # 3. Decodificar las predicciones para obtener bounding boxes
            # (en coordenadas del grid 128x128)
            detections = decode_predictions(
                hm_output[0].cpu(),
                wh_output[0].cpu(),
                reg_output[0].cpu(),
                score_threshold=0.15
            )
            
            # 4. Escalar boxes al tamaño original de la imagen
            scaled_detections = scale_boxes_to_image(
                detections, 
                (original_width, original_height)
            )
            
            # 5. Formatear respuesta con vehículos detectados
            vehicles = []
            for i, box in enumerate(scaled_detections):
                x1, y1, x2, y2, score = box
                vehicles.append({
                    'id': i + 1,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    },
                    'confidence': float(score)
                })
            
            vehicle_count = len(scaled_detections)
            print(f"Predicción: {vehicle_count} vehículos detectados.")
            
            return jsonify({
                'vehicle_count': vehicle_count,
                'vehicles': vehicles,
                'image_size': {
                    'width': original_width,
                    'height': original_height
                }
            })

        except Exception as e:
            return jsonify({'error': f'Error al procesar el archivo: {str(e)}'}), 500
    
    elif model is None:
        return jsonify({'error': 'El modelo no se ha cargado correctamente. Revisa los logs del servidor.'}), 500

    return jsonify({'error': 'Error desconocido'}), 500

if __name__ == '__main__':
    app.run(debug=True)