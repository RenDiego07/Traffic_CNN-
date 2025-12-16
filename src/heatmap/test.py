import numpy as np
import matplotlib.pyplot as plt

# --- 1. Funciones Matemáticas (Copiadas de nuestra discusión anterior) ---

def gaussian_radius(det_size, min_overlap=0.7):
    """Calcula el radio de la gaussiana basado en el tamaño del objeto (h, w)"""
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    """Genera el kernel gaussiano matemático 2D"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    """Dibuja la mancha en el heatmap in-place"""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

# --- 2. Tus Datos (Coordenadas) ---
raw_data = """
0 0.030989583333333334 0.2175925925925926 0.06197916666666667 0.1037037037037037
0 0.05130208333333333 0.28703703703703703 0.0953125 0.13333333333333333
0 0.06015625 0.7490740740740741 0.1203125 0.29814814814814816
0 0.19453125 0.4166666666666667 0.11927083333333334 0.1648148148148148
0 0.13854166666666667 0.07361111111111111 0.04895833333333333 0.06388888888888888
0 0.19609375 0.05740740740740741 0.03697916666666667 0.06666666666666667
0 0.22890625 0.040740740740740744 0.03802083333333333 0.05740740740740741
0 0.26119791666666664 0.013888888888888888 0.03802083333333333 0.027777777777777776
0 0.259375 0.05324074074074074 0.042708333333333334 0.05648148148148148
0 0.29583333333333334 0.06666666666666667 0.046875 0.06481481481481481
0 0.27890625 0.10277777777777777 0.04635416666666667 0.10185185185185185
0 0.34765625 0.021296296296296296 0.0390625 0.04259259259259259
0 0.5208333333333334 0.01574074074074074 0.03333333333333333 0.03148148148148148
0 0.43125 0.09351851851851851 0.04895833333333333 0.07777777777777778
0 0.4661458333333333 0.16342592592592592 0.071875 0.11944444444444445
2 0.9442708333333333 0.0587962962962963 0.022916666666666665 0.058333333333333334
0 0.97265625 0.5037037037037037 0.05364583333333333 0.22037037037037038
0 0.49505208333333334 0.8958333333333334 0.19739583333333333 0.20833333333333334
"""

# --- 3. Procesamiento y Generación ---

# Configuración de dimensiones
INPUT_SIZE = 512
STRIDE = 4
OUTPUT_SIZE = INPUT_SIZE // STRIDE  # 128x128
heatmap = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

lines = raw_data.strip().split('\n')
print(f"Procesando {len(lines)} objetos...")

for line in lines:
    parts = line.split()
    class_id = int(parts[0])
    
    # NOTA: Aquí podrías filtrar la clase 2 (bicicleta) si quisieras seguir 
    # la lógica estricta de "Solo Vehículos Motorizados". 
    # Para este ejemplo, graficamos todo para que veas tus datos.
    # if class_id != 0: continue

    # Coordenadas normalizadas (0-1)
    norm_cx = float(parts[1])
    norm_cy = float(parts[2])
    norm_w = float(parts[3])
    norm_h = float(parts[4])

    # Convertir a coordenadas del Grid de Salida (128x128)
    cx = norm_cx * OUTPUT_SIZE
    cy = norm_cy * OUTPUT_SIZE
    w = norm_w * OUTPUT_SIZE
    h = norm_h * OUTPUT_SIZE

    # Calcular radio adaptable
    radius = gaussian_radius((h, w))
    radius = max(0, int(radius))

    # Dibujar en el heatmap
    draw_gaussian(heatmap, np.array([cx, cy], dtype=np.int32), radius)

# --- 4. Visualización ---
plt.figure(figsize=(8, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title(f"Heatmap Ground Truth ({OUTPUT_SIZE}x{OUTPUT_SIZE})")
plt.colorbar(label='Probabilidad')
plt.axis('off')
plt.show()