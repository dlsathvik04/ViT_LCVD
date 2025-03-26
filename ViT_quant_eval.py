import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from torch.quantization import quantize_dynamic
import os

# Specify the path to the quantized model directory
model_path = 'quantized_models/model_int8'  # Using INT8 as per your error
quantization_type = "INT8"  # Set to "INT8" for this example

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
# Use CPU for INT8, GPU/MPS for FP32/FP16
model_device = cpu_device if quantization_type == "INT8" else device
print(f"Using device: {model_device}")

# Load test dataset
test_dataset = load_dataset("imagefolder", "PlantVillage", split="test")
print(f"Test dataset loaded with {len(test_dataset)} samples")

# Function to load quantized model
def load_quantized_model(model_path, quantization_type, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} not found")
    
    # Load processor
    processor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)
    
    # Load model configuration explicitly from config.json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    config = ViTConfig.from_pretrained(config_path)
    
    # Initialize model with the configuration
    model = ViTForImageClassification(config)
    
    # Load state dict from pytorch_model.pt
    state_dict_path = os.path.join(model_path, "pytorch_model.pt")
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"State dict file {state_dict_path} not found")
    state_dict = torch.load(state_dict_path, map_location=device)
    
    # Apply quantization or precision adjustments *before* loading state dict for INT8
    if quantization_type == "INT8":
        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(state_dict)  # Load after quantization
    else:
        model.load_state_dict(state_dict)  # Load first for FP32/FP16
        if quantization_type == "FP16":
            model = model.half()
    
    model.to(device)
    return model, processor

# Prediction function
def get_predictions(model, processor, dataset):
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for sample in dataset:
            inputs = processor(sample['image'], return_tensors="pt").to(model.device)
            logits = model(**inputs).logits
            pred_label = logits.argmax(-1).item()
            
            true_labels.append(sample['label'])
            pred_labels.append(pred_label)
    
    return true_labels, pred_labels

# Load model and processor
print(f"Loading {quantization_type} model from {model_path}...")
model, processor = load_quantized_model(model_path, quantization_type, model_device)

# Get predictions
true_labels, pred_labels = get_predictions(model, processor, test_dataset)
label_names = test_dataset.features['label'].names

# Calculate accuracy
accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)

# Print classification report
print(f"\nEvaluation for {quantization_type} model:")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(true_labels, pred_labels, target_names=label_names))

# Plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title(f"Confusion Matrix - {quantization_type}")
plt.savefig(f"{quantization_type.lower().replace(' ', '_')}_confusion_matrix.png")
plt.show()
print(f"Confusion matrix saved as {quantization_type.lower().replace(' ', '_')}_confusion_matrix.png")