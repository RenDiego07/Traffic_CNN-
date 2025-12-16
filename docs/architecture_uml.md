# Diagrama UML - TrafficQuantizerNet Architecture

## Diagrama de Clases Completo

```mermaid
classDiagram
    %% ============================================================
    %% PyTorch Base Classes
    %% ============================================================
    class Module {
        <<abstract>>
        +forward(x)
        +parameters()
        +train()
        +eval()
    }
    
    class Dataset {
        <<abstract>>
        +__len__()
        +__getitem__(idx)
    }
    
    %% ============================================================
    %% DATASET LAYER
    %% ============================================================
    class TrafficFlowDataset {
        -str img_dir
        -str label_dir
        -int input_size
        -int stride
        -int output_size
        -list img_files
        -list motorized_ids
        +__init__(img_dir, label_dir, input_size, stride)
        +__len__() int
        +__getitem__(idx) dict
        -_load_image(path) PIL.Image
        -_load_labels(path) list
        -_generate_targets(boxes) tuple
    }
    
    %% ============================================================
    %% MODEL ARCHITECTURE
    %% ============================================================
    class ResidualBlock {
        -Conv2d conv1
        -BatchNorm2d bn1
        -ReLU relu
        -Conv2d conv2
        -BatchNorm2d bn2
        -Sequential downsample
        +__init__(in_channels, out_channels, stride, downsample)
        +forward(x) Tensor
    }
    
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
        -_make_layer(block, channels, blocks, stride) Sequential
    }
    
    class TrafficQuantizerNet {
        -ResNetBackbone backbone
        -Sequential upsample
        -Sequential head_hm
        -Sequential head_wh
        -Sequential head_off
        +__init__()
        +forward(x) tuple~Tensor, Tensor, Tensor~
    }
    
    %% ============================================================
    %% LOSS FUNCTION
    %% ============================================================
    class TrafficLoss {
        -L1Loss l1_loss
        +__init__()
        +forward(pred_hm, pred_wh, pred_reg, batch) tuple
        -_focal_loss(pred, gt) Tensor
        -_regression_loss(pred, gt, mask) Tensor
    }
    
    %% ============================================================
    %% UTILITIES
    %% ============================================================
    class GaussianUtils {
        <<utility>>
        +gaussian_radius(det_size, min_overlap) float
        +gaussian2D(shape, sigma) ndarray
        +draw_gaussian(heatmap, center, radius, k) ndarray
    }
    
    %% ============================================================
    %% TRAINING
    %% ============================================================
    class TrainingPipeline {
        <<control>>
        -TrafficQuantizerNet model
        -TrafficLoss criterion
        -Adam optimizer
        -DataLoader train_loader
        -device device
        +train() void
        -_train_epoch(epoch) float
        -_save_checkpoint(epoch, path) void
    }
    
    %% ============================================================
    %% RELATIONSHIPS
    %% ============================================================
    
    %% Inheritance
    Dataset <|-- TrafficFlowDataset
    Module <|-- ResidualBlock
    Module <|-- ResNetBackbone
    Module <|-- TrafficQuantizerNet
    Module <|-- TrafficLoss
    
    %% Composition (Strong ownership)
    TrafficQuantizerNet *-- ResNetBackbone : contains
    TrafficQuantizerNet *-- "3" Sequential : heads
    ResNetBackbone *-- "4" Sequential : layers
    Sequential *-- "*" ResidualBlock : contains
    
    %% Aggregation (Weak ownership)
    TrainingPipeline o-- TrafficQuantizerNet : uses
    TrainingPipeline o-- TrafficLoss : uses
    TrainingPipeline o-- DataLoader : uses
    DataLoader o-- TrafficFlowDataset : wraps
    
    %% Dependencies
    TrafficFlowDataset ..> GaussianUtils : uses
    TrainingPipeline ..> TrafficFlowDataset : loads data
    TrafficLoss ..> TrafficQuantizerNet : evaluates
    
    %% Notes
    note for TrafficFlowDataset "Input: YOLO labels (normalized)\nOutput: Dense targets (128x128)\n- Heatmap (1 ch)\n- Size (2 ch)\n- Offset (2 ch)\n- Mask (1 ch)"
    
    note for TrafficQuantizerNet "CenterNet-style architecture\nInput: 512x512 RGB\nOutput: 128x128 predictions\nBackbone: ResNet-18\nParams: ~11M"
    
    note for TrafficLoss "Multi-task loss:\n1. Focal Loss (Heatmap)\n2. L1 Loss (Size)\n3. L1 Loss (Offset)\nWeights: 1.0 + 0.1 + 1.0"
```

## Diagrama de Flujo de Datos

```mermaid
flowchart TD
    %% Input Layer
    A[Raw Image<br/>Variable Size] --> B[TrafficFlowDataset]
    A1[YOLO Labels<br/>.txt files] --> B
    
    %% Dataset Processing
    B -->|Resize 512x512<br/>Normalize| C[Image Tensor<br/>3 x 512 x 512]
    B -->|Generate Targets| D[Ground Truth<br/>128 x 128]
    
    %% Ground Truth Components
    D --> D1[Heatmap<br/>1 channel]
    D --> D2[Size W,H<br/>2 channels]
    D --> D3[Offset dx,dy<br/>2 channels]
    D --> D4[Mask<br/>1 channel]
    
    %% Model Forward Pass
    C --> E[TrafficQuantizerNet]
    
    %% Backbone
    E --> F[ResNet-18 Backbone<br/>Encoder]
    F -->|512 x 16 x 16| G[Decoder<br/>Upsampling]
    
    %% Decoder to Heads
    G -->|64 x 128 x 128| H[Parallel Heads]
    
    %% Three Heads
    H --> I1[Head HM<br/>Sigmoid]
    H --> I2[Head WH<br/>Linear]
    H --> I3[Head Off<br/>Linear]
    
    %% Predictions
    I1 --> J1[Pred Heatmap<br/>1 x 128 x 128]
    I2 --> J2[Pred Size<br/>2 x 128 x 128]
    I3 --> J3[Pred Offset<br/>2 x 128 x 128]
    
    %% Loss Calculation
    J1 --> K[TrafficLoss]
    J2 --> K
    J3 --> K
    D1 --> K
    D2 --> K
    D3 --> K
    D4 --> K
    
    %% Loss Components
    K --> L1[Focal Loss<br/>Heatmap]
    K --> L2[L1 Loss<br/>Size]
    K --> L3[L1 Loss<br/>Offset]
    
    %% Total Loss
    L1 --> M[Total Loss<br/>Weighted Sum]
    L2 --> M
    L3 --> M
    
    %% Backpropagation
    M --> N[Adam Optimizer<br/>Backprop]
    N --> E
    
    %% Styling
    classDef input fill:#e1f5ff,stroke:#0066cc
    classDef processing fill:#fff4e1,stroke:#cc8800
    classDef model fill:#e8f5e9,stroke:#2e7d32
    classDef loss fill:#fce4ec,stroke:#c2185b
    classDef output fill:#f3e5f5,stroke:#7b1fa2
    
    class A,A1 input
    class B,C,D processing
    class E,F,G,H model
    class K,L1,L2,L3,M loss
    class J1,J2,J3 output
```

## Diagrama de Secuencia - Training Loop

```mermaid
sequenceDiagram
    participant Main as train()
    participant Loader as DataLoader
    participant DS as TrafficFlowDataset
    participant Model as TrafficQuantizerNet
    participant Loss as TrafficLoss
    participant Opt as Adam Optimizer
    
    Main->>Loader: Inicializar con dataset
    
    loop Por cada época
        loop Por cada batch
            Loader->>DS: __getitem__(idx)
            DS->>DS: Cargar imagen
            DS->>DS: Generar targets gaussianos
            DS-->>Loader: {input, hm, wh, reg, mask}
            Loader-->>Main: Batch de datos
            
            Main->>Model: forward(images)
            Model->>Model: Backbone (ResNet-18)
            Model->>Model: Decoder (Upsample)
            Model->>Model: 3 Heads paralelos
            Model-->>Main: (pred_hm, pred_wh, pred_off)
            
            Main->>Loss: forward(preds, targets)
            Loss->>Loss: Calcular Focal Loss
            Loss->>Loss: Calcular L1 Size Loss
            Loss->>Loss: Calcular L1 Offset Loss
            Loss-->>Main: (total_loss, l_hm, l_wh, l_off)
            
            Main->>Opt: zero_grad()
            Main->>Main: loss.backward()
            Main->>Opt: step()
            
            alt Cada 10 steps
                Main->>Main: Log métricas
            end
        end
        
        alt Cada 5 épocas
            Main->>Main: Guardar checkpoint
        end
    end
    
    Main->>Main: Guardar modelo final
```

## Arquitectura del Modelo (Detallada)

```mermaid
graph TB
    subgraph Input
        I[RGB Image<br/>3 x 512 x 512]
    end
    
    subgraph "Backbone: ResNet-18"
        S[Stem Layer<br/>Conv7x7 s=2 + MaxPool]
        L1[Layer 1<br/>2x ResBlock<br/>64 ch, 128x128]
        L2[Layer 2<br/>2x ResBlock<br/>128 ch, 64x64]
        L3[Layer 3<br/>2x ResBlock<br/>256 ch, 32x32]
        L4[Layer 4<br/>2x ResBlock<br/>512 ch, 16x16]
    end
    
    subgraph Decoder
        U1[ConvTranspose2d<br/>512→256, 32x32]
        U2[ConvTranspose2d<br/>256→128, 64x64]
        U3[ConvTranspose2d<br/>128→64, 128x128]
    end
    
    subgraph "Prediction Heads"
        H1[Heatmap Head<br/>Conv3x3 + Conv1x1<br/>→ 1 channel]
        H2[Size Head<br/>Conv3x3 + Conv1x1<br/>→ 2 channels]
        H3[Offset Head<br/>Conv3x3 + Conv1x1<br/>→ 2 channels]
    end
    
    subgraph Outputs
        O1[Heatmap<br/>1 x 128 x 128<br/>Sigmoid]
        O2[Size W,H<br/>2 x 128 x 128<br/>Linear]
        O3[Offset dx,dy<br/>2 x 128 x 128<br/>Linear]
    end
    
    I --> S
    S --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> U1
    U1 --> U2
    U2 --> U3
    U3 --> H1
    U3 --> H2
    U3 --> H3
    H1 --> O1
    H2 --> O2
    H3 --> O3
    
    style I fill:#e1f5ff
    style O1 fill:#fce4ec
    style O2 fill:#fce4ec
    style O3 fill:#fce4ec
```

## Componentes Clave

### 1. **TrafficFlowDataset**
- **Propósito**: Cargar imágenes y convertir anotaciones YOLO a targets densos
- **Entrada**: Imágenes + archivos .txt (YOLO format)
- **Salida**: Tensores con heatmaps gaussianos
- **Dependencias**: `GaussianUtils` para generar distribuciones

### 2. **TrafficQuantizerNet**
- **Propósito**: Modelo de detección end-to-end
- **Arquitectura**: Encoder-Decoder con múltiples cabezales
- **Backbone**: ResNet-18 (4 capas, 2 bloques c/u)
- **Decoder**: 3 capas de ConvTranspose2d
- **Heads**: 3 ramas paralelas para diferentes predicciones

### 3. **TrafficLoss**
- **Propósito**: Función de pérdida multi-tarea
- **Componentes**:
  - Modified Focal Loss (α=2, β=4)
  - L1 Loss para regresión de tamaño
  - L1 Loss para regresión de offset
- **Ponderación**: 1.0 : 0.1 : 1.0

### 4. **GaussianUtils**
- **Propósito**: Generar representaciones gaussianas de objetos
- **Funciones**:
  - `gaussian_radius()`: Calcula radio adaptativo
  - `gaussian2D()`: Genera kernel gaussiano
  - `draw_gaussian()`: Dibuja en heatmap con max-pooling

## Parámetros del Modelo

| Componente | Parámetros | Salida |
|------------|-----------|--------|
| **Stem** | ~9K | 64 x 128 x 128 |
| **Layer 1** | ~148K | 64 x 128 x 128 |
| **Layer 2** | ~525K | 128 x 64 x 64 |
| **Layer 3** | ~2.1M | 256 x 32 x 32 |
| **Layer 4** | ~8.4M | 512 x 16 x 16 |
| **Decoder** | ~3.5M | 64 x 128 x 128 |
| **Heads** | ~40K | 5 x 128 x 128 |
| **TOTAL** | **~11.4M** | - |

