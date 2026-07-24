import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from onnxruntime.quantization import quantize_static, CalibrationDataReader, quantize_dynamic, QuantType
import onnxruntime as ort
import time

class ADE20KcalibrationReader(CalibrationDataReader):
    def __init__(self, data_dir, num_images=100):
        self.images_dir = os.path.join(data_dir, 'images', 'validation')
        self.images = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith('.jpg')
        ])[:num_images]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.index=0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        
        img_path = os.path.join(self.images_dir, self.images[self.index])
        image    = Image.open(img_path).convert('RGB')
        image    = self.transform(image)
        image    = image.unsqueeze(0).numpy()

        self.index += 1
        if self.index % 10 == 0:
            print(f"Calibrating... {self.index}/100")

        return {"pixel_values": image}
    
ROOT      = os.path.expanduser('~/mediseg/data/ADEChallengeData2016')
fp32_path = os.path.expanduser('~/mediseg/models/segformer_b2.onnx')
int8_path = os.path.expanduser('~/mediseg/models/segformer_b2_int8.onnx')

print("starting the quantization and calibration")
reader = ADE20KcalibrationReader(ROOT, num_images=100)

quantize_static(
    model_input=fp32_path,
    model_output=int8_path,
    calibration_data_reader=reader,
    weight_type=QuantType.QInt8
)

print(f"Quantized model saved to {int8_path}")

dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)
N     = 50

# FP32 ONNX
sess_fp32 = ort.InferenceSession(fp32_path)
start = time.perf_counter()
for _ in range(N):
    sess_fp32.run(None, {"pixel_values": dummy})
fp32_ms = (time.perf_counter() - start) / N * 1000

# INT8 ONNX
sess_int8 = ort.InferenceSession(int8_path)
start = time.perf_counter()
for _ in range(N):
    sess_int8.run(None, {"pixel_values": dummy})
int8_ms = (time.perf_counter() - start) / N * 1000

print(f"\n── Quantization benchmark ──")
print(f"FP32 ONNX:  {fp32_ms:.1f} ms  ({1000/fp32_ms:.1f} FPS)")
print(f"INT8 ONNX:  {int8_ms:.1f} ms  ({1000/int8_ms:.1f} FPS)")
print(f"Speedup:    {fp32_ms/int8_ms:.2f}×")


fp32_size = os.path.getsize(fp32_path) / 1e6
int8_size = os.path.getsize(int8_path) / 1e6
print(f"\nFP32 size:  {fp32_size:.1f} MB")
print(f"INT8 size:  {int8_size:.1f} MB")
print(f"Compression: {fp32_size/int8_size:.2f}×")