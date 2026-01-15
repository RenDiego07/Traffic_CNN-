"""
Script para generar curvas de aprendizaje desde los checkpoints guardados

Extrae las métricas de loss de cada checkpoint y calcula el loss de validación
real evaluando el modelo en el conjunto de validación.

Uso:
    python scripts/generate_curves_from_checkpoints.py
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

from infer.evaluate_model import plot_learning_curves
from dataset.dtset import TrafficFlowDataset
from models.architecture import TrafficQuantizerNet
from models.loss import TrafficLoss


def extract_checkpoint_loss(checkpoint_path, device='cpu'):
    """
    Extrae el loss de un checkpoint.
    
    Args:
        checkpoint_path: Ruta al checkpoint
        device: Dispositivo donde cargar
        
    Returns:
        tuple: (epoch, loss)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        return epoch, loss
    except Exception as e:
        print(f"⚠️  Error cargando {os.path.basename(checkpoint_path)}: {e}")
        return None, None


def evaluate_on_validation(checkpoint_path, val_loader, device='cpu'):
    """
    Evalúa un checkpoint en el conjunto de validación.
    
    Args:
        checkpoint_path: Ruta al checkpoint
        val_loader: DataLoader de validación
        device: Dispositivo de cómputo
        
    Returns:
        float: Loss promedio en validación
    """
    try:
        # Cargar modelo
        model = TrafficQuantizerNet().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Criterio de pérdida
        criterion = TrafficLoss()
        
        # Evaluar
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                
                # Predicción
                hm_pred, wh_pred, off_pred = model(inputs)
                
                # Calcular loss
                loss, _, _, _ = criterion(hm_pred, wh_pred, off_pred, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
        
    except Exception as e:
        print(f"⚠️  Error evaluando {os.path.basename(checkpoint_path)}: {e}")
        return None


def collect_all_checkpoints(models_dir):
    """
    Recolecta y ordena todos los checkpoints.
    
    Args:
        models_dir: Directorio con los checkpoints
        
    Returns:
        list: Lista ordenada de rutas de checkpoints
    """
    # Buscar checkpoints con épocas
    checkpoint_pattern = os.path.join(models_dir, 'traffic_model_ep*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("❌ No se encontraron checkpoints")
        return []
    
    # Extraer número de época para ordenar
    def get_epoch_number(path):
        basename = os.path.basename(path)
        epoch_str = basename.replace('traffic_model_ep', '').replace('.pth', '')
        try:
            return int(epoch_str)
        except ValueError:
            return 999999
    
    # Ordenar por época
    checkpoints.sort(key=get_epoch_number)
    
    return checkpoints


def main():
    """Función principal"""
    print("="*70)
    print("  GENERACIÓN DE CURVAS DE APRENDIZAJE CON VALIDACIÓN REAL")
    print("="*70)
    
    # Configurar dispositivo
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🖥️  Usando Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n🖥️  Usando NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("\n🖥️  Usando CPU")
    
    # Directorio de modelos
    models_dir = os.path.join(root_dir, 'models')
    
    print(f"\n📂 Buscando checkpoints en: {models_dir}")
    
    # Recolectar checkpoints
    checkpoints = collect_all_checkpoints(models_dir)
    
    if not checkpoints:
        print("\n❌ No se encontraron checkpoints para procesar")
        return
    
    print(f"✅ Encontrados {len(checkpoints)} checkpoints")
    print(f"   Desde: {os.path.basename(checkpoints[0])}")
    print(f"   Hasta: {os.path.basename(checkpoints[-1])}")
    
    # Cargar dataset de validación
    print("\n📊 Cargando conjunto de validación...")
    data_root = os.path.join(root_dir, 'data', 'Intersection-Flow-5K')
    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_label_dir = os.path.join(data_root, 'labels', 'val')
    
    if not os.path.exists(val_img_dir):
        print(f"❌ No se encuentra el directorio de validación: {val_img_dir}")
        return
    
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
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )
    
    print(f"✅ Dataset de validación cargado: {len(val_dataset)} imágenes")
    
    # Extraer métricas
    print(f"\n📈 Procesando {len(checkpoints)} checkpoints...")
    print("   (Esto puede tomar varios minutos)")
    
    epochs = []
    train_losses = []
    val_losses = []
    
    for ckpt_path in tqdm(checkpoints, desc="Evaluando checkpoints"):
        # Extraer train loss del checkpoint
        epoch, train_loss = extract_checkpoint_loss(ckpt_path, device)
        
        if epoch is None or train_loss is None:
            continue
        
        # Calcular val loss evaluando en el conjunto de validación
        val_loss = evaluate_on_validation(ckpt_path, val_loader, device)
        
        if val_loss is None:
            continue
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    if not train_losses:
        print("\n❌ No se pudieron extraer métricas de los checkpoints")
        return
    
    print(f"\n✅ Métricas extraídas exitosamente")
    print(f"   Épocas procesadas: {len(epochs)}")
    print(f"   Primera época: {epochs[0]}")
    print(f"   Última época: {epochs[-1]}")
    print(f"\n📊 Train Loss:")
    print(f"   Inicial: {train_losses[0]:.4f}")
    print(f"   Final: {train_losses[-1]:.4f}")
    print(f"   Reducción: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    print(f"\n📊 Validation Loss:")
    print(f"   Inicial: {val_losses[0]:.4f}")
    print(f"   Final: {val_losses[-1]:.4f}")
    print(f"   Reducción: {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.1f}%")
    
    # Calcular gap entre train y val
    final_gap = ((val_losses[-1] - train_losses[-1]) / train_losses[-1]) * 100
    print(f"\n📈 Gap Train-Val (época final): {final_gap:.1f}%")
    
    # Generar gráfico
    print("\n🎨 Generando curvas de aprendizaje...")
    
    output_path = os.path.join(root_dir, 'learning_curves_real.png')
    
    plot_learning_curves(train_losses, val_losses, save_path=output_path)
    
    print("\n" + "="*70)
    print("  ✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\n📊 Gráfico guardado en: {output_path}")
    
    # Interpretación de resultados
    print("\n" + "─"*70)
    print("📝 INTERPRETACIÓN DE RESULTADOS:")
    print("─"*70)
    print("✓ Train Loss: Datos reales extraídos de los checkpoints")
    print("✓ Val Loss: Evaluado en el conjunto de validación")
    
    if final_gap < 15:
        print("\n✅ Modelo bien generalizado (gap < 15%)")
    elif final_gap < 25:
        print("\n⚠️  Ligero overfitting (gap 15-25%)")
    else:
        print("\n❌ Overfitting significativo (gap > 25%)")
    
    print("="*70)


if __name__ == "__main__":
    main()
