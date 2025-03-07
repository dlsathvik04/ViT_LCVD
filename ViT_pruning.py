import torch
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import prune

# Create pruned_models directory if it doesn't exist
output_dir = "./pruned_models"
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

# Function to apply pruning to the model
def apply_pruning(model, amount):
    # Clone the model to avoid modifying the original
    pruned_model = ViTForImageClassification.from_pretrained(model_path)
    pruned_model.to(device)
    pruned_model.eval()
    # Apply unstructured L1 pruning to Linear layers
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return pruned_model

# Store results
pruning_levels = ["0% (Original)", "20%", "40%", "60%"]
pruning_amounts = [0.0, 0.2, 0.4, 0.6]
accuracies = []
sizes = []

# Evaluate and save models at different pruning levels
for level, amount in zip(pruning_levels, pruning_amounts):
    print(f"\nProcessing pruning level: {level}")
    if amount == 0.0:
        # Original model (no pruning)
        pruned_model = model
    else:
        # Apply pruning
        pruned_model = apply_pruning(model, amount)
    
    # Evaluate
    accuracy = evaluate_model(pruned_model)
    size = get_model_size(pruned_model, f"model_pruned_{level.replace(' ', '_')}")
    accuracies.append(accuracy)
    sizes.append(size)
    
    # Save the pruned model
    save_path = os.path.join(output_dir, f"model_pruned_{amount}")
    pruned_model.save_pretrained(save_path)
    print(f"Pruned model ({level}) saved to {save_path}")

# Print results
print("\nPruning Results:")
print(f"{'Level':<15} {'Accuracy':<10} {'Size (MB)':<10}")
print("-" * 35)
for level, acc, size in zip(pruning_levels, accuracies, sizes):
    print(f"{level:<15} {acc:<10.4f} {size:<10.2f}")

# Write results to file
with open("pruning_results.txt", "w") as f:
    f.write("Pruning Results:\n")
    f.write(f"{'Level':<15} {'Accuracy':<10} {'Size (MB)':<10}\n")
    f.write("-" * 35 + "\n")
    for level, acc, size in zip(pruning_levels, accuracies, sizes):
        f.write(f"{level:<15} {acc:<10.4f} {size:<10.2f}\n")
print("Results written to pruning_results.txt")

# Plot Accuracy vs Size
plt.figure(figsize=(8, 6))
plt.scatter(sizes, accuracies, color='blue', label='Pruning Levels')
for i, level in enumerate(pruning_levels):
    plt.annotate(level, (sizes[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Model Size for Different Pruning Levels")
plt.grid(True)
plt.legend()
plt.savefig("pruning_plot.png")
plt.show()
print("Plot saved as pruning_plot.png")