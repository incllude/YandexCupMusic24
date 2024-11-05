import torch.nn.functional as F
import torch.nn as nn
import torch
import timm


class ArcFace(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_classes: int,
                 margin: float,
                 scale: float):
        super(ArcFace, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weights = nn.Parameter(torch.zeros(num_classes, emb_dim).float())
        nn.init.xavier_normal_(self.weights)
        
    def forward(self, features, targets):
        
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weights), bias=None)
        logits = cos_theta * self.scale
        
        return logits
    
    
class CNNModel(nn.Module):
    
    def __init__(self, timm_model, dropout, fusion=nn.Identity()):
        super(CNNModel, self).__init__()
        
        self.sequential = nn.Sequential(nn.BatchNorm2d(1),
                                        timm.create_model(timm_model, pretrained=True, in_chans=1, drop_rate=dropout),
                                        fusion)
        try:
            self.sequential[1].head.fc = nn.Identity()
        except:
            self.sequential[1].fc = nn.Identity()
        
    def forward(self, x):
        return self.sequential(x)
