import torch
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.quantization import quantize_dynamic

# Create quantized_models directory if it doesn't exist
output_dir = "./quantized_models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Directory already exists: {output_dir}")

# Load the test dataset
dataset_test = load_dataset("imagefolder", "PlantVillage", split="test")
print(f"Test dataset loaded with {len(dataset_test)} samples")

# Load the feature extractor
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTImageProcessor.from_pretrained(model_id)
print("Feature extractor loaded")

# Preprocessing function (same as training)
def preprocess(batch):
    images = [img.convert('RGB') if img.mode != 'RGB' else img for img in batch['image']]
    inputs = feature_extractor(images, return_tensors='pt')
    inputs['label'] = batch['label']
    return inputs

# Collate function (same as training)
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

# Prepare test dataset
prepared_test = dataset_test.with_transform(preprocess)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load the trained model
model_path = "./plantvillage_final_model_balanced"
model = ViTForImageClassification.from_pretrained(model_path)
model.to(device)
print(f"Model loaded from {model_path}")

# Training arguments for evaluation (minimal setup)
eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    logging_dir='./logs',
    remove_unused_columns=False,
)

# Function to get model size in MB
def get_model_size(model, model_name="model"):
    torch.save(model.state_dict(), f"{model_name}.pt")
    size_mb = os.path.getsize(f"{model_name}.pt") / (1024 * 1024)  # Convert to MB
    os.remove(f"{model_name}.pt")
    return size_mb

# Function to evaluate model and return accuracy
def evaluate_model(model):
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collate_fn,
        eval_dataset=prepared_test,
        tokenizer=feature_extractor,
    )
    eval_results = trainer.evaluate()
    return eval_results['eval_accuracy']

# Store results
quantization_levels = ["FP32 (Original)", "FP16", "INT8"]
accuracies = []
sizes = []

# 1. Original FP32 model
print("Evaluating FP32 model...")
model_fp32 = model
model_fp32.eval()
accuracy_fp32 = evaluate_model(model_fp32)
size_fp32 = get_model_size(model_fp32, "model_fp32")
accuracies.append(accuracy_fp32)
sizes.append(size_fp32)
model_fp32.save_pretrained(os.path.join(output_dir, "model_fp32"))
print("FP32 model saved")

# 2. FP16 model
print("Converting to FP16 and evaluating...")
model_fp16 = ViTForImageClassification.from_pretrained(model_path).half()
model_fp16.to(device)
model_fp16.eval()
accuracy_fp16 = evaluate_model(model_fp16)
size_fp16 = get_model_size(model_fp16, "model_fp16")
accuracies.append(accuracy_fp16)
sizes.append(size_fp16)
model_fp16.save_pretrained(os.path.join(output_dir, "model_fp16"))
print("FP16 model saved")

# 3. INT8 model (Dynamic Quantization)
print("Converting to INT8 and evaluating...")
model_int8 = ViTForImageClassification.from_pretrained(model_path)
model_int8.eval()
model_int8 = quantize_dynamic(model_int8, {torch.nn.Linear}, dtype=torch.qint8)
model_int8.to(device)
accuracy_int8 = evaluate_model(model_int8)
size_int8 = get_model_size(model_int8, "model_int8")
accuracies.append(accuracy_int8)
sizes.append(size_int8)
model_int8.save_pretrained(os.path.join(output_dir, "model_int8"))
print("INT8 model saved")

# Print results
print("\nQuantization Results:")
print(f"{'Level':<15} {'Accuracy':<10} {'Size (MB)':<10}")
print("-" * 35)
for level, acc, size in zip(quantization_levels, accuracies, sizes):
    print(f"{level:<15} {acc:<10.4f} {size:<10.2f}")

# Write results to file
with open("quantization_results.txt", "w") as f:
    f.write("Quantization Results:\n")
    f.write(f"{'Level':<15} {'Accuracy':<10} {'Size (MB)':<10}\n")
    f.write("-" * 35 + "\n")
    for level, acc, size in zip(quantization_levels, accuracies, sizes):
        f.write(f"{level:<15} {acc:<10.4f} {size:<10.2f}\n")
print("Results written to quantization_results.txt")

# Plot Accuracy vs Size
plt.figure(figsize=(8, 6))
plt.scatter(sizes, accuracies, color='blue', label='Quantization Levels')
for i, level in enumerate(quantization_levels):
    plt.annotate(level, (sizes[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Model Size for Different Quantization Levels")
plt.grid(True)
plt.legend()
plt.savefig("quantization_plot.png")
plt.show()
print("Plot saved as quantization_plot.png")