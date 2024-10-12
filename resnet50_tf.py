import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


import tensorflow as tf

# 사전 학습된 ResNet50 모델 로드 (ImageNet 데이터셋 기준)
model = tf.keras.applications.ResNet50(weights='imagenet')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 모델을 파일로 저장
with open('resnet50.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite 모델 저장 완료: resnet50.tflite")