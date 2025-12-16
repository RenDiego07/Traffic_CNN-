# Grid-Based Object Detection for Traffic Flow

Este proyecto implementa un sistema de detección de objetos basado en grid (similar a YOLO) para detectar vehículos en intersecciones de tráfico.

## 🎯 Concepto: Grid Detection

El modelo divide la imagen en un **grid de 16x16 celdas**. Cada celda es responsable de predecir:

1. **Objectness** (1 valor): Probabilidad de que haya un objeto en esa celda
2. **Bounding Box** (4 valores):
   - `x_offset`: Desplazamiento horizontal del centro dentro de la celda [0-1]
   - `y_offset`: Desplazamiento vertical del centro dentro de la celda [0-1]
   - `width`: Ancho del objeto (normalizado respecto a la imagen)
   - `height`: Alto del objeto (normalizado respecto a la imagen)
3. **Clase** (5 valores): Probabilidades para cada clase de vehículo

### Ejemplo Visual

```
Imagen 512x512 → Grid 16x16 (cada celda cubre 32x32 píxeles)

┌─────┬─────┬─────┬─────┐
│     │     │  🚗 │     │  ← Celda (2,2) detecta un coche
├─────┼─────┼─────┼─────┤
│     │     │     │     │
├─────┼─────┼─────┼─────┤
│     │  🚌 │     │     │  ← Celda (1,2) detecta un bus
└─────┴─────┴─────┴─────┘
```

## 📁 Estructura del Proyecto

```
src/
├── dataset/
│   └── dtset.py              # Dataset que carga imágenes y labels YOLO
├── models/
│   └── cnn_simple.py         # Modelo CNN + funciones de conversión y loss
├── train/
│   └── train_grid_detector.py # Script de entrenamiento
├── infer/
│   └── infer_grid_detector.py # Script de inferencia
notebooks/
└── visualize_grid_targets.py # Visualización del grid
```

## 🔧 Componentes Principales

### 1. Modelo CNN (`GridDetectionCNN`)

Arquitectura:
- **Encoder**: 5 bloques convolucionales que reducen 512x512 → 16x16
- **Detection Head**: Predice para cada celda del grid

```python
Input:  [B, 3, 512, 512]        # Imágenes RGB
Output: 
  - objectness: [B, 16, 16, 1]  # Heatmap de presencia de objetos
  - bbox:       [B, 16, 16, 4]  # Coordenadas de bounding boxes
  - classes:    [B, 16, 16, 5]  # Clasificación de vehículos
```

### 2. Conversión YOLO → Grid Targets (`yolo_to_grid_targets`)

Convierte las anotaciones YOLO en targets para entrenamiento:

```python
# YOLO format (normalizado)
x_center, y_center, width, height = 0.25, 0.40, 0.10, 0.20

# Conversión a grid
cell_x = int(0.25 * 16) = 4
cell_y = int(0.40 * 16) = 6

# Targets
obj_target[6, 4] = 1.0                                    # Marcar celda
bbox_target[6, 4] = [0.00, 0.40, 0.10, 0.20]              # Offsets + tamaño
class_target[6, 4] = 2                                    # Clase del vehículo
```

### 3. Función de Pérdida (`compute_loss`)

Combina múltiples pérdidas ponderadas:

```python
Total Loss = λ_coord × Loss_bbox + Loss_obj + λ_noobj × Loss_noobj + Loss_class

Donde:
- Loss_bbox:  MSE para coordenadas (solo en celdas con objetos)
- Loss_obj:   BCE para presencia de objetos
- Loss_noobj: BCE para celdas vacías (peso reducido)
- Loss_class: Cross-entropy para clasificación
```

**Pesos:**
- `λ_coord = 5.0`: Penaliza más los errores en localización
- `λ_noobj = 0.5`: Reduce la influencia de celdas vacías (son mayoría)

## 🚀 Uso

### Visualizar el Grid

Primero, visualiza cómo se convierten los labels YOLO a grid targets:

```bash
cd notebooks
python visualize_grid_targets.py
```

Esto genera `outputs/grid_visualization.png` mostrando:
- Imagen original con bounding boxes YOLO
- Heatmap de objectness
- Grid con offsets detallados

### Entrenar el Modelo

```bash
cd src/train
python train_grid_detector.py
```

**Parámetros configurables:**
```python
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3
GRID_SIZE = 16
NUM_CLASSES = 5
```

El entrenamiento guarda:
- `models/checkpoints/best_model.pth`: Mejor modelo basado en validation loss
- `models/checkpoints/model_epoch_X.pth`: Checkpoints cada 10 épocas

### Inferencia

```bash
cd src/infer
python infer_grid_detector.py
```

Genera detecciones con visualización en `outputs/detection_result.png`.

## 📊 Clases de Vehículos

```python
0: "car"         # Automóvil
1: "truck"       # Camión
2: "bus"         # Autobús
3: "motorcycle"  # Motocicleta
4: "bicycle"     # Bicicleta
```

## 🎓 Conceptos Clave

### ¿Por qué Grid Detection?

1. **Eficiencia**: Una sola pasada por la red predice todos los objetos
2. **Localización espacial**: Cada celda es responsable de su región
3. **Escalabilidad**: Fácil de adaptar a diferentes resoluciones de grid

### Comparación con YOLO Original

| Aspecto | Este Proyecto | YOLO v1 |
|---------|--------------|---------|
| Grid | 16×16 | 7×7 o 13×13 |
| Anchors | No usa | No usa (v1) |
| Backbone | CNN simple | Darknet |
| Clases | 5 | 20/80 |

### Limitaciones Actuales

1. **Una detección por celda**: Si hay múltiples objetos en la misma celda, solo detecta uno
2. **Objetos pequeños**: El grid 16×16 puede no capturar objetos muy pequeños
3. **Sin Non-Maximum Suppression (NMS)**: Pueden haber detecciones duplicadas en celdas adyacentes

### Posibles Mejoras

- [ ] Implementar múltiples anchor boxes por celda
- [ ] Añadir NMS para eliminar detecciones duplicadas
- [ ] Usar Feature Pyramid Network para detectar objetos a múltiples escalas
- [ ] Data augmentation (rotaciones, cambios de color, etc.)
- [ ] Backbone pre-entrenado (ResNet, EfficientNet)

## 📈 Monitoreo del Entrenamiento

Durante el entrenamiento, observa:

```
Train Loss: 0.8234
  - BBox: 0.1245    # ← Debe bajar (mejor localización)
  - Obj: 0.3421     # ← Debe bajar (mejor detección)
  - NoObj: 0.2156   # ← Debe bajar (menos falsos positivos)
  - Class: 0.1412   # ← Debe bajar (mejor clasificación)
```

**Señales de buen entrenamiento:**
- Training loss baja consistentemente
- Validation loss sigue al training loss (sin overfitting)
- BBox loss < 0.1 indica buena localización

## 🛠️ Troubleshooting

**Problema**: Loss muy alto al inicio
- **Solución**: Normal, espera 5-10 épocas para que converja

**Problema**: Validation loss sube mientras training loss baja
- **Solución**: Overfitting → reduce learning rate o añade dropout

**Problema**: No detecta objetos pequeños
- **Solución**: Aumenta grid_size a 32×32 o 64×64

**Problema**: Múltiples detecciones del mismo objeto
- **Solución**: Implementa NMS en el script de inferencia

## 📚 Referencias

- [YOLO v1 Paper](https://arxiv.org/abs/1506.02640)
- [Understanding YOLO](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)
- [Grid-based Detection Explained](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006)

## 🙏 Créditos

Dataset: [Intersection-Flow-5K](https://github.com/yourusername/intersection-flow-5k)
