from flask import Flask, render_template, request, jsonify
import os
from PIL import Image

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

    if file:
        try:
            img = Image.open(file.stream)
            if img.width != 1920 or img.height != 1080:
                return jsonify({'error': f'Tamaño de imagen incorrecto. Se esperaba 1920x1080, pero se obtuvo {img.width}x{img.height}.'}), 400
            
            # ------------------------------------------------------------------
            # AQUÍ SE DEBE INSERTAR LA LÓGICA DE INFERENCIA DEL MODELO
            # ------------------------------------------------------------------
            # Ejemplo de cómo se podría llamar al modelo:
            # 1. Guardar la imagen temporalmente si es necesario.
            #    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            #    file.save(image_path)
            # 2. Cargar y preprocesar la imagen según los requerimientos del modelo.
            # 3. Realizar la predicción.
            #    vehicle_count =  mi_modelo_cnn.predict(imagen_preprocesada)
            # ------------------------------------------------------------------

            # Por ahora, devolvemos un conteo de demostración.
            dummy_vehicle_count = 42
            
            return jsonify({'vehicle_count': dummy_vehicle_count})

        except Exception as e:
            return jsonify({'error': f'Error al procesar el archivo: {e}'}), 500

    return jsonify({'error': 'Error desconocido'}), 500

if __name__ == '__main__':
    app.run(debug=True)