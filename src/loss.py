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
        텐서 객체 내부적으로 채널(feature map) 별 병렬 처리를 지원하므로, 개별 feature map에 따로 접근하지 않음
        Parameters:
            x (torch.Tensor): 생성된 이미지 텐서
            y (torch.Tensor): 콘텐츠 이미지 텐서

        Returns:
            torch.Tensor: 계산된 콘텐츠 손실
            (값은 단일 스칼라지만, 역전파를 위한 계산 그래프 연결을 유지해야하기 때문에 텐서 객체로 반환)
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

        입력 텐서는 여러 개의 feature map을 포함하고 있으며,
        PyTorch는 이러한 채널 간 연산을 텐서 단위로 병렬 처리할 수 있으므로,
        각 feature map에 따로 접근하지 않고도 채널 간 상관관계를 계산하여
        스타일 정보를 표현하는 Gram 행렬을 생성할 수 있음

        Parameters:
            x (torch.Tensor): 입력 텐서, shape (b, c, h, w)
                - b: 배치 크기 (이미지 수, 1)
                - c: 채널 수 (= feature map 수)
                - h, w: 각 feature map의 세로, 가로 크기

        Returns:
            torch.Tensor: 계산된 Gram 행렬, shape (b, c, c)
                - 각 행렬은 (채널 간 내적 결과)로 구성되며, 이미지 스타일을 표현
                - 전치 행렬곱을 통해 구함
        """
        b, c, h, w = x.size()

        # feature map 펼치기: (b, c, h*w)
        # 한 채널 내 모든 공간 위치를 하나의 열 벡터로 변환
        features = x.view(b, c, h * w)

        # transpose: (b, h*w, c)
        # 채널 간 내적을 계산하기 위해 feature 벡터를 전치함
        features_T = features.transpose(1, 2)

        # batch-wise 행렬곱: (b, c, c)
        # 각 이미지에 대해 채널(feature map) 간의 내적을 계산하여 Gram 행렬 생성
        G = torch.matmul(features, features_T)

        # 정규화: 논문 정의에 따라, 스타일 손실 계산 시 Gram 행렬의 값 범위를 맞추기 위해
        # 전체 요소 수 (채널 수 × 위치 수)에 해당하는 b * c * h * w 로 나눔
        return G.div(b * c * h * w)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        생성 이미지와 x와 스타일 이미지 y의 Gram 행렬을 구한 뒤 MSE 손실 계산
        텐서 객체 내부적으로 채널(feature map) 별 병렬 처리를 지원하므로, 개별 feature map에 따로 접근하지 않음
        Parameters:
            x (torch.Tensor): 생성 이미지 텐서
            y (torch.Tensor): 스타일 이미지 텐서

        Returns:
            torch.Tensor: 계산된 스타일 손실
            (값은 단일 스칼라지만, 역전파를 위한 계산 그래프 연결을 유지해야하기 때문에 텐서 객체로 반환)
        """
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(Gx, Gy)
        return loss