"""
TrafficQuantizerNet - Arquitectura CenterNet para Detección de Vehículos

Arquitectura Anchor-Free basada en ResNet-18:
    1. Backbone: ResNet-18 modificado (encoder)
    2. Decoder: Upsampling progresivo (16x16 → 128x128)
    3. Heads: 3 ramas paralelas (Heatmap, Size, Offset)

Input:  RGB 512x512
Output: Dense predictions 128x128:
    - Heatmap (1 canal): Probabilidad de centro de objeto [0-1]
    - Size (2 canales): Width, Height en píxeles del grid
    - Offset (2 canales): Corrección subpíxel dx, dy

Inspirado en:
    - CenterNet: "Objects as Points" (Zhou et al., 2019)
    - ResNet: "Deep Residual Learning" (He et al., 2015)
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights

# --- BLOQUE RESIDUAL (La pieza fundamental) ---
class ResidualBlock(nn.Module):
    """
    Bloque residual básico de ResNet.
    
    Estructura:
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
                                              ↑
                                          identity
    
    Args:
        in_channels: Canales de entrada
        out_channels: Canales de salida
        stride: Stride para downsampling (1 o 2)
        downsample: Capa para ajustar dimensiones de skip connection
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Skip connection
        out = self.relu(out)
        return out

# --- BACKBONE (ResNet-18 Encoder) ---
class ResNetBackbone(nn.Module):
    """
    Backbone ResNet-18 modificado para extracción de features.
    
    Flujo de resolución espacial (input 512x512):
        Input: 512x512
        ↓ Stem (Conv7x7 s=2 + MaxPool s=2)
        Layer1 (64 ch):  128x128
        Layer2 (128 ch): 64x64
        Layer3 (256 ch): 32x32
        Layer4 (512 ch): 16x16 ← Salida del backbone
    
    Args:
        block: Tipo de bloque residual (ResidualBlock)
        layers: Lista con cantidad de bloques por capa [2,2,2,2] para ResNet-18
    """
    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.in_channels = 64
        
        # Stem (Reducción inicial: 512 → 128)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Capas residuales (Layers 1-4)
        self.layer1 = self._make_layer(block, 64, layers[0])       # 128x128
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 64x64
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 32x32
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 16x16
        
        # Cargar pesos preentrenados de ImageNet
        self._load_pretrained_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Construye una capa con múltiples bloques residuales."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # Ajustar dimensiones de skip connection
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        # Primer bloque (puede hacer downsampling)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        # Bloques restantes (sin downsampling)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        """
        Carga SOLO los pesos que coinciden con nuestra estructura manual.
        Ignora las capas 'fc' (Fully Connected) de la ResNet original.
        """
        print("📥 Cargando pesos de ImageNet en arquitectura manual...")
        try:
            # 1. Descargar los pesos oficiales
            weights = ResNet18_Weights.IMAGENET1K_V1
            state_dict = weights.get_state_dict(progress=True)
            
            # 2. Cargar en nuestro modelo
            # strict=False es la clave: Permite cargar los pesos coincidentes (conv1, layer1...)
            # e ignorar que nos falta la capa 'fc' y 'avgpool' que borramos a propósito.
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            print("✅ Pesos de ResNet cargados exitosamente.")
            # Verificamos que lo importante se cargó (layer1..4)
            # fc y avgpool deben aparecer en 'unexpected_keys' del state_dict original, lo cual es correcto.
            
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}")
            print("   -> Usando inicialización aleatoria.")
    
    def forward(self, x):
        """
        Forward pass del backbone.
        
        Args:
            x: Tensor (B, 3, 512, 512)
        
        Returns:
            Tensor (B, 512, 16, 16) - Features de alto nivel
        """
        x = self.conv1(x)      # 512 → 256
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 256 → 128
        x = self.layer1(x)     # 128 (64 ch)
        x = self.layer2(x)     # 64 (128 ch)
        x = self.layer3(x)     # 32 (256 ch)
        x = self.layer4(x)     # 16 (512 ch)
        return x

# --- MODELO COMPLETO (CenterNet-ResNet) ---
class TrafficQuantizerNet(nn.Module):
    """
    Modelo completo para detección de vehículos tipo CenterNet.
    
    Arquitectura:
        Input (512x512) → Backbone (ResNet-18) → Decoder (Upsample) → Heads
                                    ↓                     ↓              ↓
                                 16x16 (512ch)        128x128 (64ch)  3 outputs
    
    Outputs:
        hm: Heatmap (B, 1, 128, 128) - Probabilidad de centro de vehículo
        wh: Size (B, 2, 128, 128) - Ancho y alto del bbox
        off: Offset (B, 2, 128, 128) - Corrección de discretización
    """
    def __init__(self):
        super(TrafficQuantizerNet, self).__init__()
        
        # 1. Backbone ResNet-18 ([2,2,2,2] bloques por capa)
        self.backbone = ResNetBackbone(ResidualBlock, [2, 2, 2, 2])
        
        # 2. Decoder (Upsampling progresivo: 16x16 → 128x128)
        # De 512 canales bajamos gradualmente a 64 canales
        self.upsample = nn.Sequential(
            # 16x16 → 32x32
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            
            # 32x32 → 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            
            # 64x64 → 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
        )
        
        # 3. Heads (Cabezales de predicción paralelos)
        # Cada head: Conv3x3 (features) + Conv1x1 (output)
        
        # Heatmap: 1 canal (Vehículo Si/No)
        self.head_hm = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 1, 1)
        )
        
        # Size: 2 canales (Width, Height)
        self.head_wh = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 2, 1)
        )
        
        # Offset: 2 canales (dx, dy)
        self.head_off = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, x):
        """
        Forward pass completo del modelo.
        
        Args:
            x: Tensor (B, 3, 512, 512) - Imágenes RGB normalizadas
        
        Returns:
            hm: Tensor (B, 1, 128, 128) - Heatmap con sigmoid [0-1]
            wh: Tensor (B, 2, 128, 128) - Tamaños (w, h) sin activación
            off: Tensor (B, 2, 128, 128) - Offsets (dx, dy) sin activación
        """
        # 1. Extracción de features (Encoder)
        feat = self.backbone(x)  # (B, 512, 16, 16)
        
        # 2. Recuperación de resolución espacial (Decoder)
        up = self.upsample(feat)  # (B, 64, 128, 128)
        
        # 3. Predicciones paralelas (Heads)
        hm = torch.sigmoid(self.head_hm(up))  # Sigmoid para probabilidad [0-1]
        wh = self.head_wh(up)                 # Regresión lineal
        off = self.head_off(up)               # Regresión lineal
        
        return hm, wh, off


if __name__ == "__main__":
    # Test del modelo
    print("=== Prueba de TrafficQuantizerNet ===\n")
    
    model = TrafficQuantizerNet()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parámetros Totales: {total_params:,}")
    print(f"Parámetros Entrenables: {trainable_params:,}")
    
    # Test de forward pass
    dummy_input = torch.randn(2, 3, 512, 512)  # Batch de 2 imágenes
    
    with torch.no_grad():
        hm, wh, off = model(dummy_input)
    
    print(f"\nInput:  {dummy_input.shape}")
    print(f"Output Heatmap: {hm.shape} (rango: {hm.min():.3f} - {hm.max():.3f})")
    print(f"Output Size:    {wh.shape}")
    print(f"Output Offset:  {off.shape}")
    
    print("\n✅ Modelo inicializado correctamente")
