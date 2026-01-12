"""
Script para generar curvas de aprendizaje (Learning Curves)

Genera gráficos de Precisión y Pérdida durante el entrenamiento y validación.
Estilo profesional para publicaciones académicas.

Uso:
    python plot_learning_curves.py
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(history, save_path='learning_curves.png'):
    """
    Genera gráficos de curvas de aprendizaje con precisión y pérdida.
    
    Args:
        history: Dict con las métricas del entrenamiento. Debe contener:
            - 'train_precision': Lista de precisiones de entrenamiento por época
            - 'val_precision': Lista de precisiones de validación por época
            - 'train_loss': Lista de pérdidas de entrenamiento por época
            - 'val_loss': Lista de pérdidas de validación por época
        save_path: Ruta donde guardar la figura (default: 'learning_curves.png')
    
    Returns:
        matplotlib.figure.Figure: La figura generada
    """
    # Crear figura con 2 subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Número de épocas
    epochs = range(1, len(history['train_precision']) + 1)
    
    # ========================================================================
    # GRÁFICO IZQUIERDO: PRECISIÓN
    # ========================================================================
    ax1 = axes[0]
    
    # Líneas de entrenamiento y validación
    ax1.plot(epochs, history['train_precision'], 
             color='#1f77b4', linewidth=2, label='Entrenamiento', marker='o', 
             markersize=4, markevery=max(1, len(epochs)//10))
    
    ax1.plot(epochs, history['val_precision'], 
             color='#ff7f0e', linewidth=2, label='Validación', marker='s', 
             markersize=4, markevery=max(1, len(epochs)//10))
    
    # Configuración del gráfico
    ax1.set_xlabel('Época', fontsize=12, fontweight='normal')
    ax1.set_ylabel('Precisión', fontsize=12, fontweight='normal')
    ax1.set_title('Precisión', fontsize=14, fontweight='bold', pad=15)
    
    # Límites del eje Y
    ax1.set_ylim([0.0, 1.0])
    
    # Grid sutil
    ax1.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
    
    # Leyenda
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # Estilo de ticks
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # ========================================================================
    # GRÁFICO DERECHO: PÉRDIDA
    # ========================================================================
    ax2 = axes[1]
    
    # Líneas de entrenamiento y validación
    ax2.plot(epochs, history['train_loss'], 
             color='#1f77b4', linewidth=2, label='Entrenamiento', marker='o', 
             markersize=4, markevery=max(1, len(epochs)//10))
    
    ax2.plot(epochs, history['val_loss'], 
             color='#ff7f0e', linewidth=2, label='Validación', marker='s', 
             markersize=4, markevery=max(1, len(epochs)//10))
    
    # Configuración del gráfico
    ax2.set_xlabel('Época', fontsize=12, fontweight='normal')
    ax2.set_ylabel('Pérdida', fontsize=12, fontweight='normal')
    ax2.set_title('Pérdida', fontsize=14, fontweight='bold', pad=15)
    
    # Grid sutil
    ax2.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
    
    # Leyenda
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # Estilo de ticks
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # ========================================================================
    # AJUSTES FINALES
    # ========================================================================
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Curvas de aprendizaje guardadas: {save_path}")
    
    return fig


# ============================================================================
# EJEMPLO DE USO
# ============================================================================
if __name__ == "__main__":
    # Datos de ejemplo (reemplazar con tus datos reales)
    # Puedes cargar estos datos desde un archivo JSON, CSV, o checkpoint de PyTorch
    
    # Ejemplo 1: Datos simulados
    np.random.seed(42)
    num_epochs = 20
    
    # Simular curvas de aprendizaje típicas
    train_precision = 0.45 + 0.50 * (1 - np.exp(-np.arange(num_epochs) / 3)) + np.random.normal(0, 0.02, num_epochs)
    val_precision = 0.45 + 0.48 * (1 - np.exp(-np.arange(num_epochs) / 3)) + np.random.normal(0, 0.03, num_epochs)
    
    train_loss = 1.5 * np.exp(-np.arange(num_epochs) / 4) + 0.05 + np.random.normal(0, 0.02, num_epochs)
    val_loss = 1.5 * np.exp(-np.arange(num_epochs) / 4) + 0.10 + np.random.normal(0, 0.03, num_epochs)
    
    # Asegurar que los valores sean positivos y realistas
    train_precision = np.clip(train_precision, 0, 1)
    val_precision = np.clip(val_precision, 0, 1)
    train_loss = np.clip(train_loss, 0, None)
    val_loss = np.clip(val_loss, 0, None)
    
    history_example = {
        'train_precision': train_precision.tolist(),
        'val_precision': val_precision.tolist(),
        'train_loss': train_loss.tolist(),
        'val_loss': val_loss.tolist()
    }
    
    # Generar gráfico
    plot_learning_curves(history_example, save_path='learning_curves_example.png')
    
    print("\n" + "="*60)
    print("✅ Ejemplo generado exitosamente")
    print("="*60)
    print("\n📝 INSTRUCCIONES PARA USAR CON TUS DATOS:")
    print("\n1. Organiza tus datos en un diccionario 'history' con las claves:")
    print("   - 'train_precision': Lista de precisiones de entrenamiento")
    print("   - 'val_precision': Lista de precisiones de validación")
    print("   - 'train_loss': Lista de pérdidas de entrenamiento")
    print("   - 'val_loss': Lista de pérdidas de validación")
    print("\n2. Llama a la función:")
    print("   plot_learning_curves(history, save_path='mi_grafico.png')")
    print("\n3. Ejemplo de carga desde archivo JSON:")
    print("   import json")
    print("   with open('training_history.json', 'r') as f:")
    print("       history = json.load(f)")
    print("   plot_learning_curves(history)")
    print("\n4. Ejemplo de carga desde checkpoint de PyTorch:")
    print("   import torch")
    print("   checkpoint = torch.load('model.pth')")
    print("   history = checkpoint.get('history', {})")
    print("   plot_learning_curves(history)")
    print("="*60)
