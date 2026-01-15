import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights

class ResidualBlock(nn.Module):

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
        
        out += identity
        out = self.relu(out)
        return out

class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.in_channels = 64
    
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])      
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
        
        self._load_pretrained_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):

        print("Cargando pesos de ImageNet en arquitectura manual...")
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            state_dict = weights.get_state_dict(progress=True)
            
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("Pesos de ResNet cargados exitosamente.")
        except Exception as e:
            print(f"Error cargando pesos: {e}")
    
    def forward(self, x):
        x = self.conv1(x)      
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   
        x = self.layer1(x)     
        x = self.layer2(x)    
        x = self.layer3(x)   
        x = self.layer4(x)    
        return x

class TrafficQuantizerNet(nn.Module):
    def __init__(self):
        super(TrafficQuantizerNet, self).__init__()
        
        self.backbone = ResNetBackbone(ResidualBlock, [2, 2, 2, 2])
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
        )
        
        
        self.head_hm = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 1, 1)
        )
        
        self.head_wh = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 2, 1)
        )
        
        self.head_off = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, x):

        feat = self.backbone(x)  
        
        up = self.upsample(feat)  
        
        hm = torch.sigmoid(self.head_hm(up))  
        wh = self.head_wh(up)                 
        off = self.head_off(up)               
        
        return hm, wh, off


if __name__ == "__main__":
    print("=== Prueba de TrafficQuantizerNet ===\n")
    
    model = TrafficQuantizerNet()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parámetros Totales: {total_params:,}")
    print(f"Parámetros Entrenables: {trainable_params:,}")
    
    dummy_input = torch.randn(2, 3, 512, 512)  
    
    with torch.no_grad():
        hm, wh, off = model(dummy_input)
    
    print(f"\nInput:  {dummy_input.shape}")
    print(f"Output Heatmap: {hm.shape} (rango: {hm.min():.3f} - {hm.max():.3f})")
    print(f"Output Size:    {wh.shape}")
    print(f"Output Offset:  {off.shape}")
    
    print("\nModelo inicializado correctamente")
