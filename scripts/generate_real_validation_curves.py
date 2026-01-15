"""
Script para generar curvas de aprendizaje con validation loss REAL

Evalúa cada checkpoint guardado contra el conjunto de validación real
para obtener métricas precisas de train y validation loss.

Uso:
    python scripts/generate_real_validation_curves.py
"""

import sys
import os
import torch
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

# Setup de paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # IA/
sys.path.insert(0, os.path.join(root_dir, 'src'))

from dataset.dtset import TrafficFlowDataset
from models.architecture import TrafficQuantizerNet
from models.loss import TrafficLoss
from infer.evaluate_model import plot_learning_curves


def evaluate_checkpoint_on_validation(checkpoint_path, val_loader, criterion, device='cpu'):
    """
    Evalúa un checkpoint en el conjunto de validación.
    
    Args:
        checkpoint_path: Ruta al checkpoint
        val_loader: DataLoader de validación
        criterion: Función de pérdida
        device: Dispositivo de cómputo
        
    Returns:
        tuple: (epoch, train_loss, val_loss)
    """
    try:
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        epoch = checkpoint.get('epoch', 0)
        train_loss = checkpoint.get('loss', 0.0)
        
        # Cargar modelo
        model = TrafficQuantizerNet().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluar en validación
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                hm_pred, wh_pred, off_pred = model(inputs)
                loss, _, _, _ = criterion(hm_pred, wh_pred, off_pred, batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        return epoch, train_loss, val_loss
        
    except Exception as e:
        print(f"⚠️  Error evaluando {os.path.basename(checkpoint_path)}: {e}")
        return None, None, None


def collect_all_checkpoints(models_dir):
    """
    Recolecta y ordena todos los checkpoints.
    
    Args:
        models_dir: Directorio con los checkpoints
        
    Returns:
        list: Lista ordenada de rutas de checkpoints
    """
    checkpoint_pattern = os.path.join(models_dir, 'traffic_model_ep*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return []
    
    def get_epoch_number(path):
        basename = os.path.basename(path)
        epoch_str = basename.replace('traffic_model_ep', '').replace('.pth', '')
        try:
            return int(epoch_str)
        except ValueError:
            return 999999
    
    checkpoints.sort(key=get_epoch_number)
    return checkpoints


def main():
    """Función principal"""
    print("="*70)
    print("  CURVAS DE APRENDIZAJE CON VALIDATION LOSS REAL")
    print("="*70)
    
    # Configurar dispositivo
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🖥️  Usando: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n🖥️  Usando: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("\n🖥️  Usando: CPU")
    
    # Configurar paths
    models_dir = os.path.join(root_dir, 'models')
    data_root = os.path.join(root_dir, 'data', 'Intersection-Flow-5K')
    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_label_dir = os.path.join(data_root, 'labels', 'val')
    
    print(f"\n📂 Directorios:")
    print(f"   Checkpoints: {models_dir}")
    print(f"   Val Images: {val_img_dir}")
    print(f"   Val Labels: {val_label_dir}")
    
    # Verificar que existan los directorios
    if not os.path.exists(val_img_dir) or not os.path.exists(val_label_dir):
        print("\n❌ Error: No se encontraron directorios de validación")
        return
    
    # Crear dataset de validación
    print("\n📊 Cargando conjunto de validación...")
    val_dataset = TrafficFlowDataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        input_size=512,
        stride=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✅ Dataset cargado: {len(val_dataset)} imágenes de validación")
    
    # Crear función de pérdida
    criterion = TrafficLoss()
    
    # Recolectar checkpoints
    print(f"\n🔍 Buscando checkpoints...")
    checkpoints = collect_all_checkpoints(models_dir)
    
    if not checkpoints:
        print("❌ No se encontraron checkpoints")
        return
    
    print(f"✅ Encontrados {len(checkpoints)} checkpoints")
    print(f"   Desde: {os.path.basename(checkpoints[0])}")
    print(f"   Hasta: {os.path.basename(checkpoints[-1])}")
    
    # Evaluar cada checkpoint
    print(f"\n🔬 Evaluando checkpoints en validación (esto puede tardar)...")
    print("   " + "─"*66)
    
    epochs = []
    train_losses = []
    val_losses = []
    
    for ckpt_path in tqdm(checkpoints, desc="Evaluando"):
        epoch, train_loss, val_loss = evaluate_checkpoint_on_validation(
            ckpt_path, val_loader, criterion, device
        )
        
        if epoch is not None:
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
    
    if not train_losses:
        print("\n❌ No se pudieron evaluar los checkpoints")
        return
    
    # Mostrar resultados
    print(f"\n" + "="*70)
    print("  📈 RESULTADOS DE LA EVALUACIÓN")
    print("="*70)
    print(f"\n✅ Checkpoints evaluados: {len(epochs)}")
    print(f"\n📊 Época {epochs[0]}:")
    print(f"   Train Loss: {train_losses[0]:.4f}")
    print(f"   Val Loss:   {val_losses[0]:.4f}")
    print(f"\n📊 Época {epochs[-1]}:")
    print(f"   Train Loss: {train_losses[-1]:.4f}")
    print(f"   Val Loss:   {val_losses[-1]:.4f}")
    
    # Calcular mejora
    train_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
    val_improvement = ((val_losses[0] - val_losses[-1]) / val_losses[0]) * 100
    
    print(f"\n📉 Reducción de pérdida:")
    print(f"   Train Loss: {train_improvement:.1f}%")
    print(f"   Val Loss:   {val_improvement:.1f}%")
    
    # Calcular gap entre train y val
    final_gap = ((val_losses[-1] - train_losses[-1]) / train_losses[-1]) * 100
    print(f"\n🎯 Gap final (Val - Train): {final_gap:.1f}%")
    
    if final_gap < 20:
        print("   ✅ Excelente - Modelo bien generalizado")
    elif final_gap < 30:
        print("   ✅ Bueno - Ligero overfitting")
    else:
        print("   ⚠️  Overfitting detectado")
    
    # Generar gráfico
    print(f"\n🎨 Generando curvas de aprendizaje...")
    output_path = os.path.join(root_dir, 'learning_curves_validation_real.png')
    
    plot_learning_curves(train_losses, val_losses, save_path=output_path)
    
    print("\n" + "="*70)
    print("  ✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\n📊 Gráfico guardado en: {output_path}")
    print("\n💡 Ahora tienes:")
    print("   • Train Loss: Datos reales de los checkpoints")
    print("   • Val Loss: Calculado evaluando en el conjunto de validación")
    print("   • Métricas 100% precisas del entrenamiento")
    print("="*70)


if __name__ == "__main__":
    main()
