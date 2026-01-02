"""
TrafficLoss - Función de pérdida multi-tarea para CenterNet

Combina 3 componentes:
1. Modified Focal Loss (Heatmap): Penaliza centros vs fondo de forma adaptativa
2. L1 Loss (Size): Regresión de dimensiones W, H del bounding box
3. L1 Loss (Offset): Regresión de corrección subpíxel dx, dy

Ponderación:
    Loss_total = L_hm + 1.0 * L_offset + 0.1 * L_size

Inspirado en:
    - CenterNet: "Objects as Points" (Zhou et al., 2019)
    - Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
"""

import torch
import torch.nn as nn


class TrafficLoss(nn.Module):
    """
    Función de pérdida para detección de vehículos tipo CenterNet.
    
    Componentes:
        - Heatmap Loss: Modified Focal Loss con alpha=2, beta=4
        - Regression Losses: L1 Loss para tamaño y offset (solo en centros de objetos)
    
    Args:
        None (configuración fija optimizada para tráfico vehicular)
    
    Returns:
        loss: Pérdida total (escalar)
        loss_hm: Componente de heatmap (para logging)
        loss_wh: Componente de tamaño (para logging)
        loss_off: Componente de offset (para logging)
    """
    
    def __init__(self):
        super(TrafficLoss, self).__init__()
        # L1 Loss para regresiones (tamaño y offset)
        # reduction='sum' porque normalizamos manualmente por num_objects
        self.l1_loss = nn.L1Loss(reduction='sum') 

    def forward(self, pred_hm, pred_wh, pred_reg, batch):
        """
        Calcula la pérdida multi-tarea.
        
        Args:
            pred_hm: Predicción de heatmap (B, 1, 128, 128) [0, 1]
            pred_wh: Predicción de tamaño (B, 2, 128, 128)
            pred_reg: Predicción de offset (B, 2, 128, 128)
            batch: Dict con ground truths:
                - 'hm': (B, 1, 128, 128) - Target heatmap
                - 'wh': (B, 2, 128, 128) - Target size
                - 'reg': (B, 2, 128, 128) - Target offset
                - 'reg_mask': (B, 128, 128) - Máscara binaria (1=objeto, 0=fondo)
        
        Returns:
            loss: Pérdida total (escalar)
            loss_hm: Componente de heatmap
            loss_wh: Componente de tamaño
            loss_off: Componente de offset
        """
        # Mover targets al mismo dispositivo que las predicciones
        device = pred_hm.device
        gt_hm = batch['hm'].to(device)
        gt_wh = batch['wh'].to(device)
        gt_reg = batch['reg'].to(device)
        mask = batch['reg_mask'].to(device)  # Máscara: 1 donde hay objeto, 0 donde no
        
        # ====================================================================
        # 1. MODIFIED FOCAL LOSS (Heatmap)
        # ====================================================================
        # Separar índices positivos (centros de objetos) y negativos (fondo)
        pos_inds = gt_hm.eq(1).float()  # Píxeles exactamente en centros
        neg_inds = gt_hm.lt(1).float()  # Todo lo demás (incluye gaussiana suavizada)

        # Peso para negativos: Beta=4 (penalización reducida cerca de los centros gaussianos)
        # Si gt_hm = 0.8 (cerca del centro), neg_weight = (1-0.8)^4 = 0.0016 (casi ignorado)
        # Si gt_hm = 0.0 (fondo puro), neg_weight = (1-0.0)^4 = 1.0 (penalización completa)
        neg_weights = torch.pow(1 - gt_hm, 4)
        
        # Clamp predicción para evitar log(0)
        pred_hm = torch.clamp(pred_hm, 1e-6, 1 - 1e-6)

        # Loss para centros (positivos): -log(p) * (1-p)^alpha
        # Alpha=2: reduce impacto de ejemplos fáciles (p cercano a 1)
        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, 2) * pos_inds
        
        # Loss para fondo (negativos): -log(1-p) * p^alpha * neg_weights
        neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, 2) * neg_weights * neg_inds

        # Normalizar por cantidad de objetos en el batch
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            # Si no hay objetos en el batch, solo pérdida de fondo
            loss_hm = -neg_loss
        else:
            loss_hm = -(pos_loss + neg_loss) / num_pos

        # ====================================================================
        # 2. REGRESSION LOSSES (Size & Offset) - Solo en centros de objetos
        # ====================================================================
        # Expandir máscara para broadcasting: (B, 128, 128) -> (B, 1, 128, 128)
        mask = mask.unsqueeze(1).float()
        
        # Peso de normalización por cantidad de objetos
        reg_weight = 1.0 / (num_pos + 1e-4)

        # L1 Loss solo donde mask == 1 (centros de objetos)
        # Multiplicar por mask hace que el loss sea 0 donde no hay objetos
        loss_off = self.l1_loss(pred_reg * mask, gt_reg * mask) * reg_weight
        loss_wh = self.l1_loss(pred_wh * mask, gt_wh * mask) * reg_weight

        # ====================================================================
        # 3. TOTAL LOSS (Ponderación multi-tarea)
        # ====================================================================
        # Pesos: HM=1.0, Offset=1.0, Size=0.1
        # Razonamiento:
        #   - Heatmap es crítico (centros correctos)
        #   - Offset importante (precisión subpíxel)
        #   - Size menos crítico (tolerancia en dimensiones)
        loss = loss_hm + 1.0 * loss_off + 0.1 * loss_wh
        
        return loss, loss_hm, loss_wh, loss_off
