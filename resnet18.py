import torch
from torchvision import models

# ResNet18 모델 로드 및 TorchScript 변환
model = models.resnet18(pretrained=True)
model.eval()

# TorchScript로 변환
scripted_model = torch.jit.script(model)

# 모델 저장
scripted_model.save("resnet18.pt")