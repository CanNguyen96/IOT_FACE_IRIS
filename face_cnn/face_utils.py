"""
Utility functions for Face ResNet18 model
"""
import torch
import torch.nn as nn
from torchvision import models

class FaceCNN_ResNet(nn.Module):
    """
    FaceCNN with ResNet18 backbone for face recognition
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

def get_face_model(num_classes, embedding_dim=512, pretrained=False, backbone='resnet18'):
    """
    Create Face ResNet18 model
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Dimension of embedding vector
        pretrained: Use ImageNet pretrained weights
        backbone: Backbone architecture (only resnet18 supported)
    
    Returns:
        FaceCNN_ResNet model
    """
    if backbone != 'resnet18':
        raise ValueError(f"Only resnet18 backbone supported, got {backbone}")
    
    return FaceCNN_ResNet(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained
    )
