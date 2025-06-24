# 입력 이미지(스타일 혹은 콘텐츠)를 VGG19에 통과시켜
# 손실 계산에 필요한 계층의 Feature Map 추출하는 클래스 정의

import torch
import torch.nn as nn
from torchvision.models import vgg19

# feature map 추출을 위해 사용할 vgg19의 conv 계층 인덱스를 딕셔너리로 정의
conv = {
    'conv1_1' : 0,  # 1번째 블록의 1번째 컨볼루션 계층 (스타일 추출용)
    'conv2_1' : 5,  # 2번째 블록의 1번째 컨볼루션 계층 (스타일 추출용)
    'conv3_1' : 10, # 3번째 블록의 1번째 컨볼루션 계층 (스타일 추출용)
    'conv4_1' : 19, # 4번째 블록의 1번째 컨볼루션 계층 (스타일 추출용)
    'conv5_1' : 28, # 5번째 블록의 1번째 컨볼루션 계층 (스타일 추출용)
    'conv4_2' : 21  # 4번째 블록의 2번째 컨볼루션 계층 (콘텐츠 추출용)
}

# VGG19를 통해 Feature Map을 추출하는 클래스
class StyleTransfer(nn.Module):
    def __init__(self, ):
        # Pytorch 제공 모듈인 nn.Module의 함수들을 사용하기 위한 초기화 메서드 호출 
        super(StyleTransfer, self).__init__()

        # 사전 학습된 VGG19 모델 로드 (ImageNet으로 학습된 특징 추출기)
        # VGG19 모델의 feature 추출 부분만 사용 (분류기인 Fully Connected 계층은 제외)
        self.vgg19_model_features = vgg19(pretrained=True).features

        # 스타일, 콘텐츠 손실 계산에 사용할 conv 계층 인덱스 리스트
        self.style_layer = [conv['conv1_1'], conv['conv2_1'], conv['conv3_1'], conv['conv4_1'], conv['conv5_1']]
        self.content_layer = [conv['conv4_2']]

    # Pytorch의 nn.Module을 상속한 클래스는 반드시 forward 함수를 정의해야 함
    # model(input)과 같은 호출 시, nn.Module 내부적으로 실행되는 self.__call__(input)이 self.foward(input)를 호출하는 구조 
    # 모델에 입력이 주어졌을 때 어떻게 처리할 지를 정의함
    # 스타일 혹은 콘텐츠 이미지 x를 vgg19 추출기에 통과시킨 후 각각의 손실 계산에 필요한 feature map을 리스트에 담아 반환함
    # 입력된 이미지를 VGG19에 통과시키며, 지정된 계층에서의 feature map을 추출하여 반환 
    def forward(self, x:torch.Tensor, mode:str):
        features = []
        # 스타일이미지인지, 콘텐츠이미지에 따라 추출할 계층의 인덱스를 선택
        if mode == 'style':
            selected_layers = self.style_layer
        elif mode == 'content':
            selected_layers = self.content_layer
        else: raise ValueError(f"Invalid mode: {mode}. Choose 'style' or 'content'.")

        # 이미지 x를 VGG19의 모든 feature 계층에 순서대로 통과시키며 업데이트
        for i in range(len(self.vgg19_model_features)):
            x = self.vgg19_model_features[i](x)
            # 지정된 계층에 도달하면, 해당 feature map을 결과 리스트에 추가
            if i in selected_layers:
                features.append(x)
        # 추출된 feature map을 담은 리스트 반환
        return features