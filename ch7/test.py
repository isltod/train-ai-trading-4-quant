import tensorflow as tf
from tensorflow.python.client import device_lib
import os

# 0번 GPU만 보이도록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 경고 메시지 1(Info), 2(Warning) 출력하지 않음
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TensorFlow version:", tf.__version__)
print(device_lib.list_local_devices())
