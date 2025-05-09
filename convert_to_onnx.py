import torch
from transformers import ViTForImageClassification

# Load your PyTorch model
model_path = "plantvillage_final_model_tiny"
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

# Define a dummy input (adjust shape based on your model, e.g., 224x224x3)
dummy_input = torch.randn(1, 3, 224, 224)  # [batch_size, channels, height, width]

# Export to ONNX
onnx_path = "vit_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=14,  # Use a recent opset for compatibility
)
print(f"Model exported to {onnx_path}")