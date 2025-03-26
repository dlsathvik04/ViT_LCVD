import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from torch.quantization import quantize_dynamic
import os

# Load dataset and processor
model_path = "./plantvillage_final_model"
processor = ViTImageProcessor.from_pretrained(model_path)
dataset_test = load_dataset("imagefolder", "PlantVillage", split="test")
print(f"Test dataset loaded with {len(dataset_test)} samples")

# Output directory
output_dir = "./quantized_models"
os.makedirs(output_dir, exist_ok=True)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
cpu_device = torch.device("cpu")

# Prediction function
def get_predictions(model, dataset):
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

# Function to get model size
def get_model_size(model, model_name="model"):
    torch.save(model.state_dict(), f"{model_name}.pt")
    size_mb = os.path.getsize(f"{model_name}.pt") / (1024 * 1024)
    os.remove(f"{model_name}.pt")
    return size_mb

quantization_levels = ["FP32 (Original)", "FP16", "INT8"]
accuracies = []
sizes = []

# 1. FP32 model
print("Evaluating FP32 model...")
model_fp32 = ViTForImageClassification.from_pretrained(model_path).to(device)
true_labels, pred_labels = get_predictions(model_fp32, dataset_test)
accuracy_fp32 = (sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels))
size_fp32 = get_model_size(model_fp32, "model_fp32")
accuracies.append(accuracy_fp32)
sizes.append(size_fp32)
torch.save(model_fp32.state_dict(), os.path.join(output_dir, "model_fp32.pt"))
print("FP32 model saved")

# 2. FP16 model
print("Converting to FP16 and evaluating...")
model_fp16 = ViTForImageClassification.from_pretrained(model_path).half().to(device)
true_labels, pred_labels = get_predictions(model_fp16, dataset_test)
accuracy_fp16 = (sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels))
size_fp16 = get_model_size(model_fp16, "model_fp16")
accuracies.append(accuracy_fp16)
sizes.append(size_fp16)
torch.save(model_fp16.state_dict(), os.path.join(output_dir, "model_fp16.pt"))
print("FP16 model saved")

# 3. INT8 model
print("Converting to INT8 and evaluating...")
model_int8 = ViTForImageClassification.from_pretrained(model_path)
model_int8 = quantize_dynamic(model_int8, {torch.nn.Linear}, dtype=torch.qint8).to(cpu_device)
true_labels, pred_labels = get_predictions(model_int8, dataset_test)
accuracy_int8 = (sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels))
size_int8 = get_model_size(model_int8, "model_int8")
accuracies.append(accuracy_int8)
sizes.append(size_int8)
torch.save(model_int8.state_dict(), os.path.join(output_dir, "model_int8.pt"))
print("INT8 model saved")

# Print results
print("\nQuantization Results:")
print(f"{'Level':<15} {'Accuracy':<10} {'Size (MB)':<10}")
print("-" * 35)
for level, acc, size in zip(quantization_levels, accuracies, sizes):
    print(f"{level:<15} {acc:<10.4f} {size:<10.2f}")
