================================================================================
                    ANALIZADOR DE TRÁFICO VEHICULAR
                         Traffic CNN Interface
================================================================================

¿QUÉ ES ESTE PROYECTO?
═════════════════════════════════════════════════════════════════════════════

Este proyecto utiliza una red neuronal convolucional (CNN) entrenada 
especializada en la detección automática de vehículos en imágenes de 
intersecciones vehiculares.

El modelo TrafficQuantizerNet está diseñado para:
  • Detectar vehículos en imágenes de tráfico
  • Proporcionar bounding boxes (rectángulos) alrededor de cada vehículo
  • Contar automáticamente la cantidad de vehículos presentes
  • Calcular la confianza de cada detección

La arquitectura se basa en CenterNet, utilizando:
  • ResNet-18 como backbone para extracción de características
  • Decodificación dense de predicciones en coordenadas de bounding boxes
  • Non-Maximum Suppression (NMS) para eliminar detecciones duplicadas


¿CÓMO FUNCIONA LA INTERFAZ?
═════════════════════════════════════════════════════════════════════════════

   La aplicación proporciona una interfaz web sencilla donde puedes:
   • Cargar una imagen desde tu computadora
   • Ver el análisis
   • Visualizar los vehículos detectados con rectángulos rojos
   • Observar la confianza (probabilidad) de cada detección




CÓMO CORRER LA INTERFAZ
═════════════════════════════════════════════════════════════════════════════

REQUISITOS PREVIOS:
  • Python 3.8 o superior instalado
  • pip (gestor de paquetes de Python)

PASO 1: Instalar dependencias
─────────────────────────────────────────────────────────────────────────────
  Abre una consola/terminal en la carpeta del proyecto y ejecuta:

    pip install -r src/interface/requirements.txt

PASO 2: Ejecutar la interfaz
─────────────────────────────────────────────────────────────────────────────
  En la misma consola, ejecuta:

    python -m src.interface.app

  Verás un mensaje similar a:
    "Running on http://127.0.0.1:5000"

PASO 3: Acceder a la interfaz
─────────────────────────────────────────────────────────────────────────────
  Abre tu navegador web (Chrome, Firefox, Edge, etc.) y ve a:

    http://localhost:5000

  Deberías ver la interfaz "Analizador de Tráfico Vehicular"

PASO 4: Usar la interfaz
─────────────────────────────────────────────────────────────────────────────
  1. Haz clic en "Seleccionar archivo" o arrastra una imagen
  2. Selecciona una imagen de tráfico 
  3. Haz clic en "Analizar Imagen"
  4. Espera a que se procese
  5. Verás la imagen con rectángulos rojos alrededor de los vehículos
  6. Se mostrará el conteo total y detalles de cada vehículo detectado

PASO 5: Detener la interfaz
─────────────────────────────────────────────────────────────────────────────
  En la consola, presiona: Ctrl+C


TIPOS DE IMÁGENES RECOMENDADAS
═════════════════════════════════════════════════════════════════════════════

El modelo está especializado en detectar vehículos en intersecciones 
vehiculares. Para obtener los mejores resultados, carga imágenes que cumplan
con las siguientes características:

IDEAL PARA ESTE MODELO:
  • Imágenes de intersecciones de tráfico
  • Vistas aéreas o de altura media de calles
  • Imágenes de cámaras de vigilancia de tráfico
  • Escenas con múltiples vehículos (autos, buses, camiones)
  • Resolución: cualquier tamaño (se ajusta automáticamente)


SOLUCIÓN DE PROBLEMAS
═════════════════════════════════════════════════════════════════════════════

PROBLEMA: "ModuleNotFoundError: No module named 'torch'"
SOLUCIÓN: Instala las dependencias
  pip install -r src/interface/requirements.txt

PROBLEMA: "Error loading model" o "FileNotFoundError"
SOLUCIÓN: Verifica que el archivo existe en:
  src/saved_models/traffic_model.pth

PROBLEMA: "Port 5000 already in use"
SOLUCIÓN: El puerto está siendo usado por otra aplicación
  • Espera unos minutos y vuelve a intentar
  • O modifica el puerto en src/interface/app.py:
    app.run(debug=True, port=5001)
  • Luego accede a http://localhost:5001

PROBLEMA: El navegador no se abre automáticamente
SOLUCIÓN: Abre manualmente
  • Copia la URL que aparece en la consola
  • Pégala en tu navegador (normalmente http://localhost:5000)

PROBLEMA: La imagen no se analiza
SOLUCIÓN: Verifica que:
  • La imagen sea en formato válido (JPG, PNG, etc.)
  • El archivo no esté corrupto
  • La consola muestra algún error específico



1. Instala dependencias: pip install -r src/interface/requirements.txt
2. Ejecuta: python -m src.interface.app
3. Abre: http://localhost:5000
4. ¡Comienza a analizar imágenes de tráfico!

Para más ayuda, consulta este README nuevamente o revisa los comentarios
en el código fuente de src/interface/app.py

================================================================================
