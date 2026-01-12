from flask import Flask, render_template, request, jsonify
import os
from PIL import Image

# --- IMPORTS PARA INTELIGENCIA ARTIFICIAL ---
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from src.models.architecture import TrafficQuantizerNet


# --------------------------------------------------------------------------
# CARGA DEL MODELO
# (Se ejecuta solo una vez cuando la aplicación inicia)
# --------------------------------------------------------------------------
print("Cargando el modelo de IA...")
model_path = 'src/saved_models/traffic_model.pth' #<- Ruta a tu modelo
model = TrafficQuantizerNet()
try:
    # Carga el diccionario de checkpoint completo.
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # Extrae el state_dict del modelo desde el diccionario de checkpoint.
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # <-- MUY IMPORTANTE: Pone el modelo en modo de evaluación.
    print("Modelo PyTorch cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo PyTorch: {e}")
    model = None


# --------------------------------------------------------------------------
# FUNCIONES DE PRE Y POST-PROCESAMIENTO
# --------------------------------------------------------------------------
def preprocess_image(img: Image.Image):
    """
    Transforma la imagen para que coincida con la entrada que espera el modelo (512x512).
    """
    # ¡¡IMPORTANTE!! Si usaste otros valores de normalización en tu entrenamiento,
    # debes reemplazarlos aquí. Estos son los estándar para ImageNet.
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # Añade la dimensión del batch (lote)

def postprocess_output(heatmap: torch.Tensor, threshold=0.5):
    """
    Cuenta los picos en el mapa de calor que superan un umbral de confianza.
    """
    # El heatmap de salida tiene forma (1, 1, 128, 128)
    # Usamos max_pool2d como una forma simple de supresión de no-máximos (NMS)
    # para asegurarnos de contar solo los picos locales.
    kernel = 3
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    
    # Mantenemos solo los píxeles que son iguales al máximo en su vecindario
    keep = (hmax == heatmap).float()
    
    # Contamos cuántos de esos picos superan el umbral de confianza.
    vehicle_count = (heatmap * keep > threshold).sum().item()
    return int(vehicle_count)


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
    Recibe una imagen, valida sus dimensiones, la procesa con el modelo 
    y devuelve el conteo de vehículos.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    if file and model is not None:
        try:
            # Abrir la imagen y asegurarse de que esté en formato RGB
            img = Image.open(file.stream).convert("RGB")
            
            if img.width != 1920 or img.height != 1080:
                return jsonify({'error': f'Tamaño de imagen incorrecto. Se esperaba 1920x1080, pero se obtuvo {img.width}x{img.height}.'}), 400
            
            # --- LÓGICA DE INFERENCIA DEL MODELO ---
            
            # 1. Pre-procesar la imagen
            preprocessed_img = preprocess_image(img)

            # 2. Realizar la predicción
            with torch.no_grad():
                hm_output, _, _ = model(preprocessed_img)

            # 3. Post-procesar la salida para obtener el conteo.
            vehicle_count = postprocess_output(hm_output, threshold=0.5)
            
            print(f"Predicción: {vehicle_count} vehículos.")
            return jsonify({'vehicle_count': vehicle_count})

        except Exception as e:
            return jsonify({'error': f'Error al procesar el archivo: {e}'}), 500
    
    elif model is None:
        return jsonify({'error': 'El modelo no se ha cargado correctamente. Revisa los logs del servidor.'}), 500

    return jsonify({'error': 'Error desconocido'}), 500

if __name__ == '__main__':
    app.run(debug=True)