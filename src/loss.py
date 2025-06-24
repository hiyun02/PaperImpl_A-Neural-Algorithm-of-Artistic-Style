# 콘텐츠 손실과 스타일 손실을 정의하는 모듈
# - 콘텐츠 손실(ContentLoss): feature map 간의 MSE (Mean Squared Error)
# - 스타일 손실(StyleLoss): Gram 행렬을 기반으로 한 feature map 간의 MSE

import torch
import torch.nn as nn
import torch.nn.functional as F

# 콘텐츠 손실 계산 클래스
class ContentLoss(nn.Module):
    def __init__(self,):
        # Pytorch 제공 모듈인 nn.Module의 함수들을 사용하기 위한 초기화 메서드 호출
        super(ContentLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        생성 이미지의 feature map 과 콘텐츠 이미지의 feature map 사이의 MSE 손실 계산

        Parameters:
            x (torch.Tensor): 생성된 이미지의 feature map
            y (torch.Tensor): 콘텐츠 이미지의 feature map

        Returns:
            torch.Tensor: 계산된 콘텐츠 손실 (스칼라 값 형태의 텐서)
        """
        loss = F.mse_loss(x, y)
        return loss


# 스타일 손실 계산 클래스
class StyleLoss(nn.Module):
    def __init__(self,):
        super(StyleLoss, self).__init__()
    
    def gram_matrix(self, x:torch.Tensor):
        """
        스타일 손실을 계산하기 위해 feature map의 Gram 행렬을 구하는 함수

        Parameters:
            x (torch.Tensor): 입력 feature map, shape (b, c, h, w)

        Returns:
            torch.Tensor: 계산된 Gram 행렬, shape (b, c, c)
        """
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        생성 이미지와 x와 스타일 이미지 y의 Gram 행렬을 구한 뒤 MSE 손실 계산
로
        Parameters:
            x (torch.Tensor): 생성 이미지의 feature map
            y (torch.Tensor): 스타일 이미지의 feature map

        Returns:
            torch.Tensor: 계산된 스타일 손실
        """
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(Gx, Gy)
        return loss