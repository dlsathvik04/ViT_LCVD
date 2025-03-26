import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from torch.nn.utils import prune
import os

# Create pruned_models directory if it doesn't exist
output_dir = "./pruned_models"
os.makedirs(output_dir, exist_ok=True)
print(f"Directory ready: {output_dir}")

# Load the test dataset
dataset_test = load_dataset("imagefolder", "PlantVillage", split="test")
print(f"Test dataset loaded with {len(dataset_test)} samples")

# Load the feature extractor from the pretrained model
model_path = "./plantvillage_final_model"
processor = ViTImageProcessor.from_pretrained(model_path)
print("Feature extractor loaded from pretrained model")

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model = ViTForImageClassification.from_pretrained(model_path)
model.to(device)
print(f"Model loaded from {model_path}")

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

# Function to get model size in MB with sparsity
def get_model_size(model, model_name="model"):
    state_dict = model.state_dict()
    torch.save(state_dict, f"{model_name}.pt")
    size_mb = os.path.getsize(f"{model_name}.pt") / (1024 * 1024)
    os.remove(f"{model_name}.pt")
    return size_mb

# Function to apply pruning and convert to sparse
def apply_pruning(model, amount):
    pruned_model = ViTForImageClassification.from_pretrained(model_path)
    pruned_model.to(device)
    pruned_model.eval()
    
    # Apply pruning to Linear layers
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    # Make pruning permanent and convert to sparse
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'weight')  # Permanently applies pruning
                # Convert to sparse and update the parameter
                sparse_weight = module.weight.data.to_sparse()
                module.weight = torch.nn.Parameter(sparse_weight)
                # Ensure bias remains dense (not pruned)
                if module.bias is not None and module.bias.is_sparse:
                    module.bias = torch.nn.Parameter(module.bias.data.to_dense())
            except ValueError:
                pass
    
    return pruned_model

# Store results
pruning_levels = ["0% (Original)", "20%", "40%", "60%"]
pruning_amounts = [0.0, 0.2, 0.4, 0.6]
accuracies = []
sizes = []

# Evaluate and save models at different pruning levels
label_names = dataset_test.features['label'].names

for level, amount in zip(pruning_levels, pruning_amounts):
    print(f"\nProcessing pruning level: {level}")
    if amount == 0.0:
        pruned_model = model
    else:
        pruned_model = apply_pruning(model, amount)
    
    # Get predictions
    true_labels, pred_labels = get_predictions(pruned_model, dataset_test)
    
    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)
    accuracies.append(accuracy)
    
    # Get model size
    size = get_model_size(pruned_model, f"model_pruned_{level.replace(' ', '_')}")
    sizes.append(size)
    
    # Generate and save classification report
    report = classification_report(true_labels, pred_labels, target_names=label_names)
    report_path = os.path.join(output_dir, f"classification_report_{level.replace(' ', '_').lower()}.txt")
    with open(report_path, "w") as f:
        f.write(f"Pruning Level: {level}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - {level}")
    cm_path = os.path.join(output_dir, f"confusion_matrix_{level.replace(' ', '_').lower()}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save the pruned model with sparse weights
    save_path = os.path.join(output_dir, f"model_pruned_{amount}")
    pruned_model.save_pretrained(save_path)
    print(f"Pruned model ({level}) saved to {save_path}")

# Print results
print("\nPruning Results:")
print(f"{'Level':<15} {'Accuracy':<10} {'Size (MB)':<10}")
print("-" * 35)
for level, acc, size in zip(pruning_levels, accuracies, sizes):
    print(f"{level:<15} {acc:<10.4f} {size:<10.2f}")

# Write summary results to file
with open(os.path.join(output_dir, "pruning_results.txt"), "w") as f:
    f.write("Pruning Results:\n")
    f.write(f"{'Level':<15} {'Accuracy':<10} {'Size (MB)':<10}\n")
    f.write("-" * 35 + "\n")
    for level, acc, size in zip(pruning_levels, accuracies, sizes):
        f.write(f"{level:<15} {acc:<10.4f} {size:<10.2f}\n")
print("Summary results written to pruning_results.txt")

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
plt.savefig(os.path.join(output_dir, "pruning_plot.png"))
plt.close()
print("Plot saved as pruning_plot.png")