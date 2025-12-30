"""
Improved Iris CNN with ResNet18 backbone
Much deeper and more powerful than the simple 3-layer CNN
"""
import torch
import torch.nn as nn
import torchvision.models as models

class IrisCNN_ResNet(nn.Module):
    """
    IrisCNN with ResNet18 backbone
    - Pretrained on ImageNet (transfer learning)
    - Fine-tuned for iris recognition
    - 512-D embedding (instead of 256-D)
    """
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for grayscale (1 channel instead of 3)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Initialize from pretrained RGB weights (average across channels)
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        
        # Custom embedding layer
        self.embedding = nn.Linear(512, embedding_dim)
        self.bn_emb = nn.BatchNorm1d(embedding_dim)
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_embedding=False):
        # ResNet forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Embedding
        emb = self.embedding(x)
        emb = self.bn_emb(emb)
        
        if return_embedding:
            return emb
        
        # Classification
        return self.classifier(emb)


class IrisCNN_MobileNet(nn.Module):
    """
    Lightweight IrisCNN with MobileNetV2 backbone
    - Faster inference
    - Smaller model size
    - Good for mobile/edge devices
    """
    def __init__(self, num_classes, embedding_dim=256, pretrained=True):
        super().__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Modify first conv for grayscale
        self.features = mobilenet.features
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        if pretrained:
            # Initialize from RGB weights
            with torch.no_grad():
                original_weight = models.mobilenet_v2(pretrained=True).features[0][0].weight
                self.features[0][0].weight.data = original_weight.data.mean(dim=1, keepdim=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_embedding=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        emb = self.embedding(x)
        
        if return_embedding:
            return emb
        
        return self.classifier(emb)


class IrisCNN_EfficientNet(nn.Module):
    """
    State-of-the-art IrisCNN with EfficientNet-B0 backbone
    - Best accuracy/efficiency trade-off
    - Compound scaling
    """
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify first conv for grayscale
        self.features = efficientnet.features
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        if pretrained:
            with torch.no_grad():
                original_weight = models.efficientnet_b0(pretrained=True).features[0][0].weight
                self.features[0][0].weight.data = original_weight.data.mean(dim=1, keepdim=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_embedding=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        emb = self.embedding(x)
        
        if return_embedding:
            return emb
        
        return self.classifier(emb)


# Factory function
def get_iris_model(backbone='resnet18', num_classes=1000, embedding_dim=512, pretrained=True):
    """
    Factory function to get iris model with specified backbone
    
    Args:
        backbone: 'simple', 'resnet18', 'mobilenet', 'efficientnet'
        num_classes: number of identities
        embedding_dim: dimension of embedding vector
        pretrained: use ImageNet pretrained weights
    
    Returns:
        IrisCNN model
    """
    if backbone == 'simple':
        return IrisCNN_Simple(num_classes)
    elif backbone == 'resnet18':
        return IrisCNN_ResNet(num_classes, embedding_dim, pretrained)
    elif backbone == 'mobilenet':
        return IrisCNN_MobileNet(num_classes, embedding_dim, pretrained)
    elif backbone == 'efficientnet':
        return IrisCNN_EfficientNet(num_classes, embedding_dim, pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


if __name__ == "__main__":
    # Test models
    import torch
    
    print("="*60)
    print("IRIS MODEL ARCHITECTURE COMPARISON")
    print("="*60)
    
    models_to_test = [
        ('simple', IrisCNN_Simple(1000)),
        ('resnet18', IrisCNN_ResNet(1000, embedding_dim=512, pretrained=False)),
        ('mobilenet', IrisCNN_MobileNet(1000, embedding_dim=256, pretrained=False)),
        ('efficientnet', IrisCNN_EfficientNet(1000, embedding_dim=512, pretrained=False))
    ]
    
    dummy_input = torch.randn(1, 1, 64, 512)  # Batch=1, Channel=1, H=64, W=512
    
    for name, model in models_to_test:
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
            embedding = model(dummy_input, return_embedding=True)
        
        print(f"\n{name.upper()}:")
        print(f"  Total params      : {total_params:,}")
        print(f"  Trainable params  : {trainable_params:,}")
        print(f"  Output shape      : {output.shape}")
        print(f"  Embedding shape   : {embedding.shape}")
        print(f"  Model size (MB)   : {total_params * 4 / 1024 / 1024:.2f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("  - ResNet18      : Best accuracy, medium size")
    print("  - MobileNet     : Fast inference, small size")
    print("  - EfficientNet  : Best accuracy/efficiency balance")
    print("  - Simple        : Fastest training, baseline")
    print("="*60)
