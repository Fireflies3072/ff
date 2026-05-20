import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGG16PerceptualLossModel(nn.Module):
    # Attributes
    imagenet_mean: torch.Tensor
    imagenet_std: torch.Tensor
    
    def __init__(self):
        super().__init__()

        # Load VGG16 model
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]
        self.slice4 = vgg[16:23]
        self.slice5 = vgg[23:30]
        
        # Freeze VGG16 model
        for param in self.parameters():
            param.requires_grad = False
            
        # Register ImageNet mean and std buffer
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_tensor(self, x):
        # Calculate L2 norm of the channel dimension and divide by it
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-10
        return x / norm

    def forward(self, pred, target):
        # Convert input image range to [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        # Normalize to ImageNet standard
        pred = (pred - self.imagenet_mean) / self.imagenet_std
        target = (target - self.imagenet_mean) / self.imagenet_std
        
        # Start pipeline extraction
        loss = 0.0
        
        # Stage 1
        h_pred, h_target = self.slice1(pred), self.slice1(target)
        loss += F.mse_loss(self._normalize_tensor(h_pred), self._normalize_tensor(h_target))
        
        # Stage 2
        h_pred, h_target = self.slice2(h_pred), self.slice2(h_target)
        loss += F.mse_loss(self._normalize_tensor(h_pred), self._normalize_tensor(h_target))
        
        # Stage 3
        h_pred, h_target = self.slice3(h_pred), self.slice3(h_target)
        loss += F.mse_loss(self._normalize_tensor(h_pred), self._normalize_tensor(h_target))
        
        # Stage 4
        h_pred, h_target = self.slice4(h_pred), self.slice4(h_target)
        loss += F.mse_loss(self._normalize_tensor(h_pred), self._normalize_tensor(h_target))
        
        # Stage 5
        h_pred, h_target = self.slice5(h_pred), self.slice5(h_target)
        loss += F.mse_loss(self._normalize_tensor(h_pred), self._normalize_tensor(h_target))
        
        return loss
