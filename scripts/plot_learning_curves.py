import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(history, save_path='learning_curves_real.png'):
    """
    Dibuja las curvas de aprendizaje basándose en el historial generado.
    """
    epochs = history['epochs']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', 
             color='#1f77b4', linewidth=2, markersize=6)
    
    ax1.set_title('Convergencia del Modelo (Loss)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Épocas', fontsize=12)
    ax1.set_ylabel('Pérdida (TrafficLoss)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(epochs, history['train_precision'], '^-', label='Test Precision', 
             color='#2ca02c', linewidth=2, markersize=6)

    ax2.set_title('Evolución de la Precisión', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Épocas', fontsize=12)
    ax2.set_ylabel('Precisión (0.0 - 1.0)', fontsize=12)
    ax2.set_ylim(0, 1.05) 
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada exitosamente en: {save_path}")