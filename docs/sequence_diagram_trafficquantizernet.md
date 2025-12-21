# Diagrama de Secuencia - TrafficQuantizerNet

## Diagrama Simplificado (Alto Nivel)

### Interacción entre Componentes Principales

```mermaid
sequenceDiagram
    actor User as Input Image<br/>(B, 3, 512, 512)
    participant TQN as TrafficQuantizerNet
    participant Backbone as ResNetBackbone
    participant ResBlock as ResidualBlock
    participant Decoder as Decoder (Upsample)
    participant Heads as PredictionHeads
    
    User->>TQN: forward(image)
    activate TQN
    
    Note over TQN,Backbone: Fase 1: Feature Extraction
    TQN->>Backbone: Extraer características
    activate Backbone
    
    loop 4 capas (layer1-4)
        Backbone->>ResBlock: Procesar features
        activate ResBlock
        Note right of ResBlock: F(x) + x<br/>Skip connection
        ResBlock-->>Backbone: Features procesadas
        deactivate ResBlock
    end
    
    Backbone-->>TQN: Features comprimidas<br/>(B, 512, 16, 16)
    deactivate Backbone
    
    Note over TQN,Decoder: Fase 2: Spatial Recovery
    TQN->>Decoder: Recuperar resolución
    activate Decoder
    Note right of Decoder: Upsample<br/>16×16 → 128×128
    Decoder-->>TQN: Features expandidas<br/>(B, 64, 128, 128)
    deactivate Decoder
    
    Note over TQN,Heads: Fase 3: Multi-task Prediction
    TQN->>Heads: Generar predicciones
    activate Heads
    
    par Predicción Paralela
        Heads->>Heads: Heatmap Head<br/>(1 canal)
    and
        Heads->>Heads: Size Head<br/>(2 canales)
    and
        Heads->>Heads: Offset Head<br/>(2 canales)
    end
    
    Heads-->>TQN: 3 Tensores de salida
    deactivate Heads
    
    TQN-->>User: (hm, wh, off)<br/>128×128 cada uno
    deactivate TQN
```

## Rol de Cada Componente

```mermaid
sequenceDiagram
    box rgb(255,240,230) Encoder
        participant BB as ResNetBackbone
        participant RB as ResidualBlock
    end
    box rgb(230,240,255) Decoder
        participant Dec as Decoder
    end
    box rgb(240,255,240) Heads
        participant PH as PredictionHeads
    end
    participant Out as Outputs
    
    Note over BB: Reduce dimensión espacial<br/>Aumenta canales semánticos
    BB->>RB: Delega procesamiento
    activate RB
    Note right of RB: • Aprende features<br/>• Skip connections<br/>• Gradientes fluidos
    RB-->>BB: Features enriquecidas
    deactivate RB
    
    Note over Dec: Recupera resolución espacial<br/>Reduce canales
    BB->>Dec: Features comprimidas<br/>512 ch @ 16×16
    activate Dec
    Note right of Dec: • ConvTranspose2d<br/>• Upsampling×8<br/>• BatchNorm + ReLU
    Dec-->>PH: Features espaciales<br/>64 ch @ 128×128
    deactivate Dec
    
    Note over PH: Genera predicciones<br/>específicas por tarea
    PH->>PH: Head 1: Heatmap
    Note right of PH: ¿Dónde hay objetos?
    PH->>PH: Head 2: Size
    Note right of PH: ¿Qué tamaño tienen?
    PH->>PH: Head 3: Offset
    Note right of PH: ¿Corrección subpíxel?
    PH-->>Out: 3 mapas densos
```

## Función de Cada Clase

| Clase | Responsabilidad | Input | Output |
|-------|----------------|-------|--------|
| **ResNetBackbone** | Extracción de características<br/>Compresión espacial | (B, 3, 512, 512) | (B, 512, 16, 16) |
| **ResidualBlock** | Transformación no lineal<br/>con skip connection | (B, C_in, H, W) | (B, C_out, H', W') |
| **Decoder** | Recuperación espacial<br/>Upsampling progresivo | (B, 512, 16, 16) | (B, 64, 128, 128) |
| **PredictionHeads** | Predicciones multi-tarea<br/>paralelas | (B, 64, 128, 128) | 3× (B, C, 128, 128) |

---

## Lógica Interna del Modelo (Detallada)

### Secuencia de Forward Pass Completo

```mermaid
sequenceDiagram
    participant Input as Input Tensor<br/>(B, 3, 512, 512)
    participant TQN as TrafficQuantizerNet
    participant BB as ResNetBackbone
    participant Stem as Stem Layer
    participant L1 as Layer1 (ResBlocks)
    participant L2 as Layer2 (ResBlocks)
    participant L3 as Layer3 (ResBlocks)
    participant L4 as Layer4 (ResBlocks)
    participant Dec as Decoder (Upsample)
    participant HM as Head Heatmap
    participant WH as Head Size
    participant OFF as Head Offset
    participant Out as Output Tuple
    
    Input->>TQN: forward(x)
    activate TQN
    
    Note over TQN: Llamada al Backbone
    TQN->>BB: backbone.forward(x)
    activate BB
    
    Note over BB,Stem: Fase 1: Reducción Inicial
    BB->>Stem: Stem(x)<br/>Conv7x7 stride=2
    Stem-->>BB: (B, 64, 256, 256)
    BB->>Stem: MaxPool stride=2
    Stem-->>BB: (B, 64, 128, 128)
    
    Note over BB,L1: Fase 2: Feature Extraction
    BB->>L1: layer1(x)
    activate L1
    loop 2 ResidualBlocks
        L1->>L1: ResBlock.forward()
        Note right of L1: F(x) + x
    end
    L1-->>BB: (B, 64, 128, 128)
    deactivate L1
    
    BB->>L2: layer2(x)
    activate L2
    loop 2 ResidualBlocks
        L2->>L2: ResBlock.forward()
        Note right of L2: Downsample: stride=2
    end
    L2-->>BB: (B, 128, 64, 64)
    deactivate L2
    
    BB->>L3: layer3(x)
    activate L3
    loop 2 ResidualBlocks
        L3->>L3: ResBlock.forward()
        Note right of L3: Downsample: stride=2
    end
    L3-->>BB: (B, 256, 32, 32)
    deactivate L3
    
    BB->>L4: layer4(x)
    activate L4
    loop 2 ResidualBlocks
        L4->>L4: ResBlock.forward()
        Note right of L4: Downsample: stride=2
    end
    L4-->>BB: (B, 512, 16, 16)
    deactivate L4
    
    BB-->>TQN: features (B, 512, 16, 16)
    deactivate BB
    
    Note over TQN,Dec: Fase 3: Decoder (Upsampling)
    TQN->>Dec: upsample(features)
    activate Dec
    Dec->>Dec: ConvTranspose2d<br/>512→256, stride=2
    Note right of Dec: (B, 256, 32, 32)
    Dec->>Dec: BatchNorm + ReLU
    Dec->>Dec: ConvTranspose2d<br/>256→128, stride=2
    Note right of Dec: (B, 128, 64, 64)
    Dec->>Dec: BatchNorm + ReLU
    Dec->>Dec: ConvTranspose2d<br/>128→64, stride=2
    Note right of Dec: (B, 64, 128, 128)
    Dec->>Dec: BatchNorm + ReLU
    Dec-->>TQN: upsampled (B, 64, 128, 128)
    deactivate Dec
    
    Note over TQN,OFF: Fase 4: Prediction Heads (Paralelos)
    
    par Head Heatmap
        TQN->>HM: head_hm(upsampled)
        activate HM
        HM->>HM: Conv3x3 (64→64)
        HM->>HM: ReLU
        HM->>HM: Conv1x1 (64→1)
        HM->>HM: Sigmoid
        HM-->>TQN: hm (B, 1, 128, 128)<br/>[0-1]
        deactivate HM
    and Head Size
        TQN->>WH: head_wh(upsampled)
        activate WH
        WH->>WH: Conv3x3 (64→64)
        WH->>WH: ReLU
        WH->>WH: Conv1x1 (64→2)
        WH-->>TQN: wh (B, 2, 128, 128)<br/>Linear
        deactivate WH
    and Head Offset
        TQN->>OFF: head_off(upsampled)
        activate OFF
        OFF->>OFF: Conv3x3 (64→64)
        OFF->>OFF: ReLU
        OFF->>OFF: Conv1x1 (64→2)
        OFF-->>TQN: off (B, 2, 128, 128)<br/>Linear
        deactivate OFF
    end
    
    Note over TQN: Combinar Outputs
    TQN->>Out: return (hm, wh, off)
    Out-->>Input: Tuple[Tensor, Tensor, Tensor]
    deactivate TQN
```

## Flujo Detallado por Componente

### 1. Backbone - ResNetBackbone

```mermaid
sequenceDiagram
    participant Input as x (B,3,512,512)
    participant BB as ResNetBackbone
    participant Conv as conv1 (Conv7x7)
    participant BN as bn1 (BatchNorm)
    participant Act as ReLU
    participant Pool as MaxPool2d
    participant L1 as layer1
    participant L2 as layer2
    participant L3 as layer3
    participant L4 as layer4
    
    Input->>BB: forward(x)
    BB->>Conv: conv1(x)
    Conv-->>BB: (B,64,256,256)
    BB->>BN: bn1(x)
    BN-->>BB: normalized
    BB->>Act: relu(x)
    Act-->>BB: activated
    BB->>Pool: maxpool(x)
    Pool-->>BB: (B,64,128,128)
    
    BB->>L1: layer1(x)
    Note over L1: 2× ResBlock<br/>64 channels<br/>No downsampling
    L1-->>BB: (B,64,128,128)
    
    BB->>L2: layer2(x)
    Note over L2: 2× ResBlock<br/>128 channels<br/>stride=2 (1st block)
    L2-->>BB: (B,128,64,64)
    
    BB->>L3: layer3(x)
    Note over L3: 2× ResBlock<br/>256 channels<br/>stride=2 (1st block)
    L3-->>BB: (B,256,32,32)
    
    BB->>L4: layer4(x)
    Note over L4: 2× ResBlock<br/>512 channels<br/>stride=2 (1st block)
    L4-->>BB: (B,512,16,16)
    
    BB-->>Input: features
```

### 2. ResidualBlock - Bloque Individual

```mermaid
sequenceDiagram
    participant Input as x (in_ch, H, W)
    participant RB as ResidualBlock
    participant Conv1 as conv1 (Conv3x3)
    participant BN1 as bn1
    participant Act1 as ReLU
    participant Conv2 as conv2 (Conv3x3)
    participant BN2 as bn2
    participant Down as downsample
    participant Add as Addition
    participant Act2 as ReLU (final)
    
    Input->>RB: forward(x)
    Note over RB: Guardar identity
    RB->>RB: identity = x
    
    Note over RB,Conv2: Main Path: F(x)
    RB->>Conv1: conv1(x)
    Conv1-->>RB: (out_ch, H', W')
    RB->>BN1: bn1(x)
    BN1-->>RB: normalized
    RB->>Act1: relu(x)
    Act1-->>RB: activated
    
    RB->>Conv2: conv2(x)
    Conv2-->>RB: (out_ch, H', W')
    RB->>BN2: bn2(x)
    BN2-->>RB: F(x) normalized
    
    alt downsample != None
        Note over RB,Down: Skip Connection Adjustment
        RB->>Down: downsample(identity)
        Down-->>RB: adjusted identity
    end
    
    Note over RB,Add: Residual Connection
    RB->>Add: out = F(x) + identity
    Add-->>RB: sum
    
    RB->>Act2: relu(out)
    Act2-->>RB: activated
    
    RB-->>Input: output
```

### 3. Decoder - Upsample Module

```mermaid
sequenceDiagram
    participant Input as features (B,512,16,16)
    participant Dec as Sequential (upsample)
    participant CT1 as ConvTranspose2d (512→256)
    participant BN1 as BatchNorm2d (256)
    participant Act1 as ReLU
    participant CT2 as ConvTranspose2d (256→128)
    participant BN2 as BatchNorm2d (128)
    participant Act2 as ReLU
    participant CT3 as ConvTranspose2d (128→64)
    participant BN3 as BatchNorm2d (64)
    participant Act3 as ReLU
    
    Input->>Dec: forward(x)
    
    Note over Dec,CT1: Stage 1: 16×16 → 32×32
    Dec->>CT1: kernel=4, stride=2, padding=1
    CT1-->>Dec: (B,256,32,32)
    Dec->>BN1: normalize
    BN1-->>Dec: normalized
    Dec->>Act1: activate
    Act1-->>Dec: (B,256,32,32)
    
    Note over Dec,CT2: Stage 2: 32×32 → 64×64
    Dec->>CT2: kernel=4, stride=2, padding=1
    CT2-->>Dec: (B,128,64,64)
    Dec->>BN2: normalize
    BN2-->>Dec: normalized
    Dec->>Act2: activate
    Act2-->>Dec: (B,128,64,64)
    
    Note over Dec,CT3: Stage 3: 64×64 → 128×128
    Dec->>CT3: kernel=4, stride=2, padding=1
    CT3-->>Dec: (B,64,128,128)
    Dec->>BN3: normalize
    BN3-->>Dec: normalized
    Dec->>Act3: activate
    Act3-->>Dec: (B,64,128,128)
    
    Dec-->>Input: upsampled features
```

### 4. Prediction Heads (Paralelos)

```mermaid
sequenceDiagram
    participant Input as upsampled (B,64,128,128)
    box rgb(255,240,240) Heatmap Head
        participant HM1 as Conv3x3 (64→64)
        participant HM2 as ReLU
        participant HM3 as Conv1x1 (64→1)
        participant HM4 as Sigmoid
    end
    box rgb(240,255,240) Size Head
        participant WH1 as Conv3x3 (64→64)
        participant WH2 as ReLU
        participant WH3 as Conv1x1 (64→2)
    end
    box rgb(240,240,255) Offset Head
        participant OFF1 as Conv3x3 (64→64)
        participant OFF2 as ReLU
        participant OFF3 as Conv1x1 (64→2)
    end
    participant Out as Outputs
    
    Note over Input: Procesamiento Paralelo
    
    par Heatmap Prediction
        Input->>HM1: x
        HM1-->>HM2: (B,64,128,128)
        HM2-->>HM3: activated
        HM3-->>HM4: (B,1,128,128)
        HM4-->>Out: hm [0-1]
    and Size Prediction
        Input->>WH1: x
        WH1-->>WH2: (B,64,128,128)
        WH2-->>WH3: activated
        WH3-->>Out: wh (B,2,128,128)
    and Offset Prediction
        Input->>OFF1: x
        OFF1-->>OFF2: (B,64,128,128)
        OFF2-->>OFF3: activated
        OFF3-->>Out: off (B,2,128,128)
    end
```

## Transformaciones de Dimensiones

### Tabla de Cambios Espaciales

| Etapa | Operación | Input Shape | Output Shape | Cambio |
|-------|-----------|-------------|--------------|--------|
| **Input** | - | (B, 3, 512, 512) | - | - |
| **Stem Conv** | Conv7×7 s=2 | (B, 3, 512, 512) | (B, 64, 256, 256) | ÷2 |
| **Stem Pool** | MaxPool s=2 | (B, 64, 256, 256) | (B, 64, 128, 128) | ÷2 |
| **Layer 1** | 2× ResBlock | (B, 64, 128, 128) | (B, 64, 128, 128) | = |
| **Layer 2** | 2× ResBlock | (B, 64, 128, 128) | (B, 128, 64, 64) | ÷2 |
| **Layer 3** | 2× ResBlock | (B, 128, 64, 64) | (B, 256, 32, 32) | ÷2 |
| **Layer 4** | 2× ResBlock | (B, 256, 32, 32) | (B, 512, 16, 16) | ÷2 |
| **Decoder 1** | ConvT 512→256 | (B, 512, 16, 16) | (B, 256, 32, 32) | ×2 |
| **Decoder 2** | ConvT 256→128 | (B, 256, 32, 32) | (B, 128, 64, 64) | ×2 |
| **Decoder 3** | ConvT 128→64 | (B, 128, 64, 64) | (B, 64, 128, 128) | ×2 |
| **Head HM** | Conv→Sigmoid | (B, 64, 128, 128) | (B, 1, 128, 128) | = |
| **Head WH** | Conv→Linear | (B, 64, 128, 128) | (B, 2, 128, 128) | = |
| **Head OFF** | Conv→Linear | (B, 64, 128, 128) | (B, 2, 128, 128) | = |

### Resumen del Flujo

```
Input: 512×512 (RGB)
    ↓ Stem (÷4)
128×128 (64 ch)
    ↓ Layer1 (=)
128×128 (64 ch)
    ↓ Layer2 (÷2)
64×64 (128 ch)
    ↓ Layer3 (÷2)
32×32 (256 ch)
    ↓ Layer4 (÷2)
16×16 (512 ch) ← Bottleneck
    ↓ Decoder (×8)
128×128 (64 ch)
    ↓ 3 Heads
128×128 (5 ch total)
    • Heatmap: 1 ch
    • Size: 2 ch
    • Offset: 2 ch
```

## Tiempo de Ejecución Estimado

| Componente | Operaciones | Tiempo Relativo |
|------------|-------------|-----------------|
| Stem | Conv7×7 + Pool | ~5% |
| Layer1-4 (Backbone) | 8 ResBlocks | ~60% |
| Decoder | 3 ConvTranspose2d | ~25% |
| Heads | 3× (Conv3×3 + Conv1×1) | ~10% |

**Total Forward Pass:** ~15-30ms en Apple M1/M2 (batch=1)
