# 입력이미지와 생성이미지를 비교하고(손실함수), 손실이 적어지도록 최적화함
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image

from src.models import StyleTransfer
from src.loss import ContentLoss, StyleLoss

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Arrow를 통한 return type hinting
def pre_processing(image:Image.Image) -> torch.Tensor:
    # 사전학습된 모델이 학습할 때 사용되었던 전처리기법을 학습 데이터에도 그대로 적용해야 함
    preprocessing = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize(mean,std) # lambda x : (x-mean) / std
    ]) # (c, h, w)

    # (1, c, h, w)
    image_tensor:torch.Tensor = preprocessing(image).unsqueeze(0)
    return image_tensor

def post_processing(tensor:torch.Tensor) -> Image.Image:
    
    # shape 1,c,h,w
    image:np.ndarray = tensor.to('cpu').detach().numpy()
    # shape c,h,w
    image = image.squeeze()
    # shape h,w,c
    image = image.transpose(1, 2, 0)
    # de norm
    image = image*std + mean
    # clip
    image = image.clip(0, 1) * 255
    # dtype unit8
    image = image.astype(np.uint8)
    # numpy -> Image
    return Image.fromarray(image)

def train_main():
    # load data
    content_image = Image.open('./img/content.jpg')
    style_image = Image.open('./img/style.jpg')

    ## pre processing
    content_image = pre_processing(content_image)
    style_image = pre_processing(style_image)

    # load model
    style_transfer = StyleTransfer()

    # load loss
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # hyper parameter
    alpha = 1
    beta = 1
    lr = 0.01

    # setting optimizer
    x = torch.randn(1, 3, 512, 512)
    optimizer = optim.Adam([x], lr=lr)

    # train loop

    ## loss print
    ## post processing
    ## image gen output save
    pass

if __name__ == "__main__":
    train_main()