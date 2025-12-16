# Diagrama de Clases UML - TrafficQuantizerNet

## Sistema de Detección de Vehículos basado en CenterNet

### Diagrama Principal (5 Clases Core)

```mermaid
classDiagram
    %% ============================================================
    %% Base Classes de PyTorch
    %% ============================================================
    class Dataset {
        <<abstract>>
        +__len__() int
        +__getitem__(index) Any
    }
    
    class Module {
        <<abstract>>
        +forward(x) Tensor
        +parameters() Iterator
        +train() Module
        +eval() Module
        +to(device) Module
    }
    
    %% ============================================================
    %% 1. TrafficFlowDataset (Preprocesamiento)
    %% ============================================================
    class TrafficFlowDataset {
        -str img_dir
        -str label_dir
        -int input_size
        -int stride
        -int output_size
        -list~str~ img_files
        -list~int~ motorized_ids
        +__init__(img_dir, label_dir, input_size, stride)
        +__len__() int
        +__getitem__(idx) dict
        -_load_image(path) PIL.Image
        -_apply_transform(img) Tensor
        -_load_labels(path) list
        -_generate_gaussian_targets(boxes) tuple
    }
    
    %% ============================================================
    %% 2. ResidualBlock (Bloque Constructivo)
    %% ============================================================
    class ResidualBlock {
        -Conv2d conv1
        -BatchNorm2d bn1
        -ReLU relu
        -Conv2d conv2
        -BatchNorm2d bn2
        -Sequential shortcut
        +__init__(in_channels, out_channels, stride, downsample)
        +forward(x) Tensor
    }
    
    %% ============================================================
    %% 3. ResNetBackbone (Extractor de Características)
    %% ============================================================
    class ResNetBackbone {
        -int in_channels
        -Conv2d conv1
        -BatchNorm2d bn1
        -ReLU relu
        -MaxPool2d maxpool
        -Sequential layer1
        -Sequential layer2
        -Sequential layer3
        -Sequential layer4
        +__init__(block, layers)
        +forward(x) Tensor
        -_make_layer(block, out_channels, blocks, stride) Sequential
    }
    
    %% ============================================================
    %% 4. TrafficQuantizerNet (Red Completa)
    %% ============================================================
    class TrafficQuantizerNet {
        -ResNetBackbone backbone
        -Sequential upsample
        -Sequential head_hm
        -Sequential head_wh
        -Sequential head_off
        +__init__()
        +forward(x) tuple~Tensor, Tensor, Tensor~
        +_load_pretrained_weights(path) void
        -_init_decoder() Sequential
        -_init_heads() void
    }
    
    %% ============================================================
    %% 5. TrafficLoss (Función de Pérdida Multi-tarea)
    %% ============================================================
    class TrafficLoss {
        -L1Loss l1_loss
        -float alpha
        -float beta
        +__init__()
        +forward(pred_hm, pred_wh, pred_reg, batch) tuple
        -_modified_focal_loss(pred, gt) Tensor
        -_l1_regression_loss(pred, gt, mask) Tensor
    }
    
    %% ============================================================
    %% RELACIONES
    %% ============================================================
    
    %% Herencia
    Dataset <|-- TrafficFlowDataset : inherits
    Module <|-- ResidualBlock : inherits
    Module <|-- ResNetBackbone : inherits
    Module <|-- TrafficQuantizerNet : inherits
    Module <|-- TrafficLoss : inherits
    
    %% Composición (TrafficQuantizerNet contiene componentes)
    TrafficQuantizerNet *-- "1" ResNetBackbone : contains
    TrafficQuantizerNet *-- "1" Sequential : upsample decoder
    TrafficQuantizerNet *-- "3" Sequential : head_hm, head_wh, head_off
    
    %% Composición (ResNetBackbone contiene bloques)
    ResNetBackbone *-- "8" ResidualBlock : contains (2x4 layers)
    
    %% Dependencia (Uso durante entrenamiento)
    TrafficQuantizerNet ..> TrafficLoss : uses during training
    TrafficFlowDataset ..> TrafficQuantizerNet : provides data
    
    %% ============================================================
    %% Notas Explicativas
    %% ============================================================
    note for TrafficFlowDataset "Rol: Preprocesamiento\n• Input: Imágenes + YOLO labels\n• Output: Dict con tensores\n• Genera targets gaussianos\n• Normaliza imágenes a 512x512"
    
    note for ResidualBlock "Rol: Bloque Constructivo\n• Implementa F(x) + x\n• Skip connection\n• 2 Conv3x3 + 2 BN + ReLU"
    
    note for ResNetBackbone "Rol: Feature Extractor\n• 4 capas (layer1-4)\n• Cada capa: 2 ResidualBlock\n• Downsampling: 512→16×16\n• Output: 512 canales"
    
    note for TrafficQuantizerNet "Rol: Red Completa (CenterNet)\n• 1 Backbone (Encoder)\n• 1 Decoder (Upsampling)\n• 3 Heads (hm, wh, off)\n• Input: 512×512 RGB\n• Output: 128×128 predictions"
    
    note for TrafficLoss "Rol: Multi-task Loss\n• Modified Focal Loss (α=2, β=4)\n• L1 Loss (Size)\n• L1 Loss (Offset)\n• Weights: 1.0 + 0.1 + 1.0"
```

## Diagrama de Composición Detallado

```mermaid
graph TB
    subgraph "TrafficQuantizerNet (Main Network)"
        TQN[TrafficQuantizerNet<br/>nn.Module]
        
        subgraph "Backbone Component"
            BB[ResNetBackbone<br/>Feature Extractor]
            
            subgraph "Layer Structure"
                L1[Layer 1: 2× ResBlock<br/>64 ch, 128×128]
                L2[Layer 2: 2× ResBlock<br/>128 ch, 64×64]
                L3[Layer 3: 2× ResBlock<br/>256 ch, 32×32]
                L4[Layer 4: 2× ResBlock<br/>512 ch, 16×16]
                
                RB1[ResidualBlock 1]
                RB2[ResidualBlock 2]
                RB3[ResidualBlock ...]
                RB8[ResidualBlock 8]
            end
        end
        
        subgraph "Decoder Component"
            UP[Upsample Module<br/>Sequential]
            U1[ConvTranspose2d: 512→256]
            U2[ConvTranspose2d: 256→128]
            U3[ConvTranspose2d: 128→64]
        end
        
        subgraph "Prediction Heads (3)"
            HHM[head_hm<br/>Conv3x3+Conv1x1<br/>→ 1 channel]
            HWH[head_wh<br/>Conv3x3+Conv1x1<br/>→ 2 channels]
            HOFF[head_off<br/>Conv3x3+Conv1x1<br/>→ 2 channels]
        end
    end
    
    TQN -->|contains| BB
    TQN -->|contains| UP
    TQN -->|contains| HHM
    TQN -->|contains| HWH
    TQN -->|contains| HOFF
    
    BB -->|contains| L1
    BB -->|contains| L2
    BB -->|contains| L3
    BB -->|contains| L4
    
    L1 -.->|uses| RB1
    L1 -.->|uses| RB2
    L4 -.->|uses| RB8
    
    UP -->|sequential| U1
    U1 -->|sequential| U2
    U2 -->|sequential| U3
    
    style TQN fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style BB fill:#fff3e0,stroke:#e65100
    style UP fill:#e1f5fe,stroke:#0277bd
    style HHM fill:#fce4ec,stroke:#c2185b
    style HWH fill:#fce4ec,stroke:#c2185b
    style HOFF fill:#fce4ec,stroke:#c2185b
```

## Tabla de Relaciones UML

| Clase Origen | Relación | Clase Destino | Tipo | Cardinalidad |
|--------------|----------|---------------|------|--------------|
| **TrafficFlowDataset** | Herencia | Dataset | Inheritance | - |
| **ResidualBlock** | Herencia | nn.Module | Inheritance | - |
| **ResNetBackbone** | Herencia | nn.Module | Inheritance | - |
| **TrafficQuantizerNet** | Herencia | nn.Module | Inheritance | - |
| **TrafficLoss** | Herencia | nn.Module | Inheritance | - |
| **TrafficQuantizerNet** | Composición | ResNetBackbone | Composition (◆) | 1 |
| **TrafficQuantizerNet** | Composición | Sequential (upsample) | Composition (◆) | 1 |
| **TrafficQuantizerNet** | Composición | Sequential (heads) | Composition (◆) | 3 |
| **ResNetBackbone** | Composición | ResidualBlock | Composition (◆) | 8 |
| **TrafficQuantizerNet** | Dependencia | TrafficLoss | Dependency (⋯>) | - |
| **TrafficFlowDataset** | Dependencia | TrafficQuantizerNet | Dependency (⋯>) | - |

## Leyenda de Símbolos UML

```
Herencia:          ───|>     (Triángulo vacío)
Composición:       ───◆      (Diamante relleno - ownership fuerte)
Agregación:        ───◇      (Diamante vacío - ownership débil)
Dependencia:       ⋯⋯⋯>     (Línea punteada - uso temporal)
Asociación:        ────      (Línea sólida - relación general)
```

## Detalles de Implementación

### 1. TrafficFlowDataset
```python
# Propósito: Cargar y preprocesar datos
# Retorna: {
#     'input': Tensor (3, 512, 512),
#     'hm': Tensor (1, 128, 128),      # Heatmap gaussiano
#     'wh': Tensor (2, 128, 128),      # Width, Height
#     'reg': Tensor (2, 128, 128),     # Offset dx, dy
#     'reg_mask': Tensor (128, 128)    # Máscara binaria
# }
```

### 2. ResidualBlock
```python
# Fórmula: output = F(x) + x
# Donde F(x) = BN(Conv(BN(Conv(x))))
# Si stride != 1: x pasa por downsample (Conv1x1)
```

### 3. ResNetBackbone
```python
# Arquitectura: ResNet-18
# Configuración: [2, 2, 2, 2] bloques por capa
# Flujo espacial: 512 → 256 → 128 → 64 → 32 → 16
# Canales: 3 → 64 → 64 → 128 → 256 → 512
```

### 4. TrafficQuantizerNet
```python
# Componentes:
# 1. backbone: ResNetBackbone (encoder)
# 2. upsample: Sequential de 3 ConvTranspose2d (decoder)
# 3. head_hm: Sequential (predicción de heatmap)
# 4. head_wh: Sequential (predicción de tamaño)
# 5. head_off: Sequential (predicción de offset)
```

### 5. TrafficLoss
```python
# Cálculo:
# loss_total = loss_hm + 1.0 * loss_off + 0.1 * loss_wh
#
# Componentes:
# - Modified Focal Loss: penaliza centros vs fondo
# - L1 Loss (Size): solo en píxeles con objetos
# - L1 Loss (Offset): solo en píxeles con objetos
```

## Flujo de Datos Completo

```
[Imagen Raw] 
    ↓
[TrafficFlowDataset]
    ↓
[Tensor 3×512×512] ──→ [TrafficQuantizerNet]
                            ↓
                        [ResNetBackbone]
                            ↓ (512×16×16)
                        [Upsample Decoder]
                            ↓ (64×128×128)
                        [3 Prediction Heads]
                            ↓
    ┌───────────────────────┼───────────────────────┐
    ↓                       ↓                       ↓
[Heatmap 1×128×128]   [Size 2×128×128]   [Offset 2×128×128]
    ↓                       ↓                       ↓
    └───────────────────────┴───────────────────────┘
                            ↓
                      [TrafficLoss]
                            ↓
                    [Loss Total Escalar]
                            ↓
                    [Backpropagation]
```

## Métricas de la Arquitectura

| Componente | Bloques | Parámetros | Dimensión Salida |
|------------|---------|------------|------------------|
| **Stem (conv1)** | 1 | 9.4K | 64 × 256 × 256 |
| **Layer 1** | 2 × ResBlock | 148K | 64 × 128 × 128 |
| **Layer 2** | 2 × ResBlock | 525K | 128 × 64 × 64 |
| **Layer 3** | 2 × ResBlock | 2.1M | 256 × 32 × 32 |
| **Layer 4** | 2 × ResBlock | 8.4M | 512 × 16 × 16 |
| **Upsample** | 3 ConvTranspose | 3.5M | 64 × 128 × 128 |
| **Heads** | 3 × Sequential | 40K | 5 × 128 × 128 |
| **TOTAL** | - | **~11.4M** | - |

