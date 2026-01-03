"""
Improved Face CNN with ResNet backbone for ArcFace training
"""
import torch
import torch.nn as nn
import torchvision.models as models

class FaceRecognitionModel(nn.Module):
    """
    Face Recognition Model with ResNet18 backbone
    Using Softmax Loss (same as Iris CNN)
    """
    def __init__(self, num_classes, embedding_size=512, pretrained=True, dropout=0.5):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract feature layers (remove final FC)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Embedding layer
        self.embedding = nn.Linear(512, embedding_size)
        self.bn_emb = nn.BatchNorm1d(embedding_size)
        
        # Classifier layer (for softmax)
        self.classifier = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass
        Returns: logits for training, or embeddings for inference
        """
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        # Embedding
        emb = self.embedding(x)
        emb = self.bn_emb(emb)
        
        if return_embedding:
            return emb
        
        # Classification
        return self.classifier(emb)


class FaceCNN_ResNet(nn.Module):
    """
    Legacy FaceCNN with ResNet18 backbone (with classifier)
    Use FaceRecognitionModel for ArcFace training
    """
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super().__init__()
        
        resnet = models.resnet18(pretrained=pretrained)
        
        # Keep RGB (3 channels) for face
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        
        # Embedding layer
        self.embedding = nn.Linear(512, embedding_dim)
        self.bn_emb = nn.BatchNorm1d(embedding_dim)
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_embedding=False):
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
        
        emb = self.embedding(x)
        emb = self.bn_emb(emb)
        
        if return_embedding:
            return emb
        
        return self.classifier(emb)


class FaceCNN_MobileNet(nn.Module):
    """Lightweight face model with MobileNetV2"""
    def __init__(self, num_classes, embedding_dim=256, pretrained=True):
        super().__init__()
        
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.embedding = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
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
def get_face_model(backbone='resnet18', num_classes=5749, embedding_dim=512, pretrained=True):
    """
    Get face model with specified backbone
    
    Args:
        backbone: 'simple', 'resnet18', 'mobilenet'
        num_classes: number of identities
        embedding_dim: dimension of embedding
        pretrained: use ImageNet pretrained weights
    """
    if backbone == 'simple':
        from face_model import FaceCNN
        return FaceCNN(num_classes)
    elif backbone == 'resnet18':
        return FaceCNN_ResNet(num_classes, embedding_dim, pretrained)
    elif backbone == 'mobilenet':
        return FaceCNN_MobileNet(num_classes, embedding_dim, pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


if __name__ == "__main__":
    print("NOTE: For production face recognition, use InsightFace (already implemented)")
    print("This is alternative if you want to train your own face model")
    
    dummy_input = torch.randn(1, 3, 112, 112)
    model = FaceCNN_ResNet(num_classes=5749, embedding_dim=512, pretrained=False)
    
    output = model(dummy_input)
    embedding = model(dummy_input, return_embedding=True)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
