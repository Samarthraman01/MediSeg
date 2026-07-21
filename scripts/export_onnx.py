import torch
from transformers import SegformerForSemanticSegmentation
import os

device = torch.device("cpu")   # export on CPU — why? see question below

# load your trained model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(
    os.path.expanduser('~/mediseg/models/segformer_b2_epoch9.pt'),
    map_location=device
))
model.eval()   # ← why is this critical before export?
dummy_input = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    model,                          # the model to export
    dummy_input,                    # fake input for tracing
    "segformer_b2.onnx",            # output filename
    input_names=["pixel_values"],   # name the input
    output_names=["logits"],        # name the output
    dynamic_axes={
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "logits":       {0: "batch", 2: "height", 3: "width"}
    },
    opset_version=14,               # ONNX operator set version
    dynamo=False                    # use the legacy TorchScript-tracing exporter (matches dynamic_axes/opset_version API above)
)