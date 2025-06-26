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
    # dtype unit8
    image = image.astype(np.uint8)
    # numpy -> Image
    return Image.fromarray(image)

def train_main():
    # load data
    content_image = Image.open('../img/content.jpg')
    style_image = Image.open('../img/style2.jpg')

    ## pre processing
    content_image = pre_processing(content_image)
    style_image = pre_processing(style_image)

    # load model
    style_transfer = StyleTransfer().eval()

    # load loss
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # hyper parameter
    alpha = 1
    beta = 1e6
    lr = 1

    save_path = f'..\\result\\{alpha}_{beta}_{lr}_initContent_style2_LBFGS'
    os.makedirs(save_path, exist_ok=True)

    # device setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"device : {device}")

    style_transfer = style_transfer.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    # noise
    # x = torch.randn(1, 3, 512, 512).to(device)
    x = content_image.clone()
    x.requires_grad_(True)

    # setting optimizer
    optimizer = optim.LBFGS([x], lr=lr)

    # closure for optimizer:LBGFS
    def closure():
        # gradient 계산 후 loss return

        optimizer.zero_grad()

        ## content representation (x, content_image)
        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')
        
        ## style representation (x, style_image)
        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')

        ## Loss_content, loss_style
        loss_c = 0
        loss_s = 0
        loss_total = 0

        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        loss_c = alpha * loss_c

        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)
        loss_s = beta * loss_s

        loss_total = loss_c + loss_s
        loss_total.backward()
        return loss_total

    # train loop
    steps = 1000
    for step in tqdm(range(steps)):    
        
        ## optimizer step
        optimizer.step(closure)

        ## loss print
        if step % 100 == 0:
            with torch.no_grad():
                
                ## content representation (x, content_image)
                x_content_list = style_transfer(x, 'content')
                y_content_list = style_transfer(content_image, 'content')
                
                ## style representation (x, style_image)
                x_style_list = style_transfer(x, 'style')
                y_style_list = style_transfer(style_image, 'style')

                ## Loss_content, loss_style
                loss_c = 0
                loss_s = 0
                loss_total = 0

                for x_content, y_content in zip(x_content_list, y_content_list):
                    loss_c += content_loss(x_content, y_content)
                loss_c = alpha * loss_c

                for x_style, y_style in zip(x_style_list, y_style_list):
                    loss_s += style_loss(x_style, y_style)
                loss_s = beta * loss_s

                loss_total = loss_c + loss_s
                        
                print(f"loss_c : {loss_c.cpu()}")
                print(f"loss_s : {loss_s.cpu()}")
                print(f"loss_total : {loss_total.cpu()}")
                
                ## post processing
                ## image gen output save
                gen_img:Image.Image = post_processing(x)
                gen_img.save(os.path.join(save_path, f'{step}.jpg'))

if __name__ == "__main__":
    train_main()