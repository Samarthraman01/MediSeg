import torch
import numpy as np
import onnxruntime as ort
from transformers import SegformerForSemanticSegmentation
import os

# ── Load PyTorch model ────────────────────────────────────────────────
print("Loading PyTorch model...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(
    os.path.expanduser('~/mediseg/models/segformer_b2_epoch9.pt'),
    map_location="cpu"
))
model.eval()

# ── Load ONNX model ───────────────────────────────────────────────────
print("Loading ONNX model...")
onnx_session = ort.InferenceSession(
    os.path.expanduser('~/mediseg/segformer_b2.onnx')
)

# ── Create one shared input ───────────────────────────────────────────
dummy = torch.randn(1, 3, 512, 512)

# ── PyTorch inference ─────────────────────────────────────────────────
print("Running PyTorch inference...")
with torch.no_grad():
    torch_out = model(pixel_values=dummy).logits
torch_out_np = torch_out.numpy()

# ── ONNX inference ────────────────────────────────────────────────────
print("Running ONNX inference...")
onnx_input = dummy.numpy()
onnx_out = onnx_session.run(
    None,                          # None = return all outputs
    {"pixel_values": onnx_input}   # feed by input name
)[0]                               # [0] = first output = logits

# ── Compare ───────────────────────────────────────────────────────────
print(f"\nPyTorch output shape: {torch_out_np.shape}")
print(f"ONNX output shape:    {onnx_out.shape}")

max_diff  = np.abs(torch_out_np - onnx_out).max()
mean_diff = np.abs(torch_out_np - onnx_out).mean()

print(f"\nMax difference:  {max_diff}")
print(f"Mean difference: {mean_diff}")

if max_diff < 1e-3:
    print("\nPASS — ONNX matches PyTorch. Export is trustworthy.")
else:
    print("\nFAIL — outputs diverge. Export is broken, do not use.")