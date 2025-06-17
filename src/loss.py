# content loss
# deep image representation -> pytorch representation
# MSE

# style loss
# gram matrix -> function
# MSE

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self,):
        super(ContentLoss, self).__init__

    # 생성이미지 feature map과 컨텐츠이미지 feature map을 입력으로 손실값을 구함
    def forward(self, x:torch.Tensor, y:torch.Tensor):
        # MSE loss
        loss = F.mse_loss(x,y)
        return loss

class StyleLoss(nn.Module):
    def __init__(self,):
        super(StyleLoss, self).__init__
    
    def gram_matrix(self, x:torch.Tensor):
        """
        x: torch.Tensor, shape (b,c,h,w)
        reshape (b,c,h,w) -> (b,c,h*w)
        dim (b, N, M)
        transpose
        matrix
        """
        b, c, h, w = x.size()
        # reshape
        features = x.view(b, c, h*w) # (b, N, M)
        features_T = features.transpose(1,2) # (b, M, N)
        G = torch.matmul(features, features_T) # (b, N, N)
        return G.div(b * c * h * w)

    def forward(self, x, y):
        # gram matrix style representation
        # MSE
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(Gx, Gy)
        return loss