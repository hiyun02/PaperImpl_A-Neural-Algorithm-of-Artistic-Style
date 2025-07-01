# 입력이미지와 생성이미지를 비교하고(손실함수), 손실이 적어지도록 최적화함
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image

from models import StyleTransfer
from loss import ContentLoss, StyleLoss

import os
from tqdm import tqdm # 진행 표시줄

# VGG19 모델 학습에 사용된 정규화 기준값
# 사전학습된 VGG19 모델이 기대하는 입력 분포에 맞추기 위해 사용
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 이미지 전처리 함수
# PIL 이미지 (h, w, c) -> torch.Tensor (b, c, h, w)
def pre_processing(image:Image.Image) -> torch.Tensor:
    # 사전학습된 모델이 학습할 때 사용되었던 전처리기법을 학습 데이터에도 그대로 적용해야 함
    preprocessing = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize(mean,std) # lambda x : (x-mean) / std
    ]) # (c, h, w)
    # 배치 차원 추가 (1, c, h, w)
    image_tensor:torch.Tensor = preprocessing(image).unsqueeze(0)
    return image_tensor

# 이미지 후처리 함수
# torch.Tensor (b, c, h, w) -> PIL 이미지 (h, w, c)
def post_processing(tensor:torch.Tensor) -> Image.Image:
    # PIL은 Pytroch 텐서를 직접 처리할 수 없기 때문에, 텐서를 먼저 Numpy 배열로 변경해야함
    # 텐서를 GPU에서 CPU로 이동시키고, 연산 그래프에서 분리(detach) 후 NumPy 배열로 변환
    image:np.ndarray = tensor.to('cpu').detach().numpy()
    # 배치 차원 제거 (c,h,w)
    image = image.squeeze()
    # PIL 이미지 형식에 맞도록 차원 순서 재배치 (h, w, c)
    image = image.transpose(1, 2, 0)
    # 정규화 복원
    # 전처리 과정에서 정규화했던 것을 되돌림
    image = image*std + mean
    # 복원 과정에서 발생한 이상값(음수 등)을 제거하기 위해 픽셀 값 범위를 [0, 255]로 되돌림
    image = image.clip(0, 1) * 255
    # 이미지 형식에 맞게 정수형(unit8)으로 변환
    image = image.astype(np.uint8)
    # numpy 배열을 PIL 이미지로 변환
    return Image.fromarray(image)

def train_main():
    # 콘텐츠, 스타일 이미지 로드
    content_image = Image.open('../img/content.jpg')
    style_image = Image.open('../img/style2.jpg')

    ## 로드한 이미지 전처리
    content_image = pre_processing(content_image)
    style_image = pre_processing(style_image)

    # 모델 및 손실함수 로드
    style_transfer = StyleTransfer().eval() # 추론/평가 전용 모드
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # 하이퍼 파라미터 설정
    alpha = 1
    beta = 1e6
    lr = 1

    # 저장 경로 설정
    save_path = f'..\\result\\{alpha}_{beta}_{lr}_initContent_style2_LBFGS'
    os.makedirs(save_path, exist_ok=True)

    # 디바이스 설정 및 이동
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"device : {device}")
    style_transfer = style_transfer.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    # 생성 이미지 정의
    # x = torch.randn(1, 3, 512, 512).to(device)
    x = content_image.clone()
    x.requires_grad_(True)

    # 손실 값을 줄이기 위한 옵티마이저 로드 (논문에서 LBFGS를 사용함)
    optimizer = optim.LBFGS([x], lr=lr)

    # LBFGS 옵티마이저가 사용할 손실 계산 함수 정의
    def compute_losses(x):
        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')
        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')

        loss_c = sum(content_loss(a, b) for a, b in zip(x_content_list, y_content_list)) * alpha
        loss_s = sum(style_loss(a, b) for a, b in zip(x_style_list, y_style_list)) * beta
        loss_total = loss_c + loss_s
        return loss_c, loss_s, loss_total

    # LBGFS optimizer 동작을 위한 손실 계산 및 역전파 기능 정의
    # 일반적인 옵티마이저는 한번의 forward -> backward -> step() 루프만으로 동작 가능하지만,
    # LBFGFS는 내부적으로 여러 번 손실 값 및 그래디언트를 재계산하여 파라미터 업데이트 방향을 추정하는 방식임
    # 따라서 아래와 같이 손실 계산 및 역전파 기능을 함수로 정의하고, 옵티마이저가 이를 활용할 수 있도록 전달함
    def closure():
        optimizer.zero_grad()
        _, _, loss_total = compute_losses(x)
        loss_total.backward()
        return loss_total

    # 학습 루프 시작
    steps = 1000
    for step in tqdm(range(steps)):    
        
        # 손실을 최소화하도록 생성 이미지(x)를 업데이트 (closure 내부에서 손실 계산 및 역전파 수행)
        optimizer.step(closure)

        # 100 step 마다 손실값 로깅 및 결과 이미지 저장
        if step % 100 == 0:
            with torch.no_grad():
                loss_c, loss_s, loss_total = compute_losses(x)
                        
                print(f"loss_c : {loss_c.cpu()}")
                print(f"loss_s : {loss_s.cpu()}")
                print(f"loss_total : {loss_total.cpu()}")
                
                # 이미지 후처리 후 지정된 경로로 저장
                gen_img:Image.Image = post_processing(x)
                gen_img.save(os.path.join(save_path, f'{step}.jpg'))

if __name__ == "__main__":
    train_main()