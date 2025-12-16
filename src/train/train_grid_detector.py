"""
Script de Entrenamiento para TrafficQuantizerNet

Sistema de detección de vehículos motorizados basado en CenterNet.

Pipeline:
    1. Carga de dataset Intersection-Flow-5K (filtrado: solo motorizados)
    2. Entrenamiento con optimizer Adam
    3. Multi-task loss (Heatmap + Size + Offset)
    4. Soporte para Apple Silicon (MPS), CUDA y CPU
    5. Guardado de checkpoints cada 5 épocas

Configuración optimizada para:
    - Hardware: Apple M1/M2/M3 con 16GB RAM
    - Batch Size: 8
    - Learning Rate: 1.25e-4
    - Épocas: 50

Uso:
    python train_grid_detector.py
    
    Importante: Ajustar las rutas img_dir y label_dir según tu estructura de carpetas
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

# --- SETUP DE IMPORTACIÓN ---
# Esto permite importar desde carpetas hermanas (dataset, models, utils)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src/
root_dir = os.path.dirname(parent_dir)     # IA/
sys.path.insert(0, parent_dir)  # Agregar src/ al path
sys.path.insert(0, root_dir)    # Agregar IA/ al path

from dataset.dtset import TrafficFlowDataset
from models.architecture import TrafficQuantizerNet
from models.loss import TrafficLoss

def train():
    """
    Función principal de entrenamiento.
    
    Flujo:
        1. Configuración de dispositivo (MPS/CUDA/CPU)
        2. Carga de datos (DataLoader)
        3. Inicialización de modelo y optimizador
        4. Loop de entrenamiento
        5. Guardado de checkpoints
    """
    
    # ========================================================================
    # 1. CONFIGURACIÓN DE DISPOSITIVO
    # ========================================================================
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Entrenando en Apple M1/M2/M3 (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Entrenando en NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️  Entrenando en CPU (Lento)")

    # ========================================================================
    # 2. HIPERPARÁMETROS
    # ========================================================================
    BATCH_SIZE = 8        # Ajustar según memoria disponible (8 para 16GB RAM)
    LEARNING_RATE = 1.25e-4
    NUM_EPOCHS = 50
    NUM_WORKERS = 2       # Para carga de datos paralela
    
    print(f"\n📊 Configuración:")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Épocas: {NUM_EPOCHS}")
    print(f"   - Workers: {NUM_WORKERS}")
    
    # ========================================================================
    # 3. CARGA DE DATOS
    # ========================================================================
    # IMPORTANTE: Cambia estas rutas según tu estructura de carpetas
    # Por defecto busca en: IA/data/Intersection-Flow-5K/images/train
    #                       IA/data/Intersection-Flow-5K/labels/train
    
    data_root = os.path.join(root_dir, 'data', 'Intersection-Flow-5K')
    train_img_dir = os.path.join(data_root, 'images', 'train')
    train_label_dir = os.path.join(data_root, 'labels', 'train')
    
    # Verificar que existan las carpetas
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"❌ No se encuentra la carpeta de imágenes: {train_img_dir}")
    if not os.path.exists(train_label_dir):
        raise FileNotFoundError(f"❌ No se encuentra la carpeta de labels: {train_label_dir}")
    
    print(f"\n📂 Rutas de datos:")
    print(f"   - Imágenes: {train_img_dir}")
    print(f"   - Labels: {train_label_dir}")
    
    train_dataset = TrafficFlowDataset(
        img_dir=train_img_dir, 
        label_dir=train_label_dir,
        input_size=512,
        stride=4
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")  # Acelerar transferencias GPU
    )
    
    print(f"\n✅ Dataset cargado: {len(train_dataset)} imágenes")
    print(f"   - Batches por época: {len(train_loader)}")

    # ========================================================================
    # 4. MODELO Y OPTIMIZADOR
    # ========================================================================
    model = TrafficQuantizerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = TrafficLoss()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modelo inicializado")
    print(f"   - Parámetros: {total_params:,}")
    print(f"   - Dispositivo: {device}")

    # Crear carpeta para guardar checkpoints
    checkpoint_dir = os.path.join(root_dir, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================================
    # 5. BUCLE DE ENTRENAMIENTO
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"INICIANDO ENTRENAMIENTO")
    print(f"{'='*60}\n")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_loss_hm = 0
        epoch_loss_wh = 0
        epoch_loss_off = 0
        
        for i, batch in enumerate(train_loader):
            # Mover imágenes a GPU/MPS
            inputs = batch['input'].to(device)
            
            # Forward pass
            hm_pred, wh_pred, off_pred = model(inputs)
            
            # Calcular Loss
            loss, l_hm, l_wh, l_off = criterion(hm_pred, wh_pred, off_pred, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Acumular métricas
            epoch_loss += loss.item()
            epoch_loss_hm += l_hm.item()
            epoch_loss_wh += l_wh.item()
            epoch_loss_off += l_off.item()
            
            # Logging cada 10 steps
            if i % 10 == 0:
                print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] Step [{i:3d}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (HM:{l_hm.item():.3f} WH:{l_wh.item():.3f} Off:{l_off.item():.3f})")
        
        # Promedios de la época
        avg_loss = epoch_loss / len(train_loader)
        avg_loss_hm = epoch_loss_hm / len(train_loader)
        avg_loss_wh = epoch_loss_wh / len(train_loader)
        avg_loss_off = epoch_loss_off / len(train_loader)
        
        print(f"\n{'─'*60}")
        print(f"📈 Epoch {epoch+1} Completada")
        print(f"   Loss Promedio: {avg_loss:.4f}")
        print(f"   - Heatmap:  {avg_loss_hm:.4f}")
        print(f"   - Size:     {avg_loss_wh:.4f}")
        print(f"   - Offset:   {avg_loss_off:.4f}")
        print(f"{'─'*60}\n")
        
        # Guardar checkpoint cada 5 épocas
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"traffic_model_ep{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint guardado: {checkpoint_path}\n")

    # ========================================================================
    # 6. FINALIZACIÓN
    # ========================================================================
    # Guardar modelo final
    final_path = os.path.join(checkpoint_dir, "traffic_model_final.pth")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"💾 Modelo final guardado: {final_path}")
    print(f"📁 Checkpoints en: {checkpoint_dir}")


if __name__ == "__main__":
    train()
