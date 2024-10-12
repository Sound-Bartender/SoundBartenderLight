import torch
import torchvision.models as models

# ResNet50 모델 불러오기 (사전 학습된 모델)
model = models.resnet50(pretrained=True)
model.eval()

# 샘플 입력 텐서
example_input = torch.rand(1, 3, 224, 224)

# 모델을 TorchScript 형식으로 변환 (트레이스 모드)
traced_model = torch.jit.trace(model, example_input)

# TorchScript 모델 저장
traced_model.save("resnet50_traced_model.pt")