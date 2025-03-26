import torch
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import os
import numpy as np
import matplotlib.pyplot as plt

# Create distilled_models directory if it doesn't exist
output_dir = "./distilled_models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Directory already exists: {output_dir}")

# Load the dataset
dataset_train = load_dataset("imagefolder", data_dir="PlantVillage", split="train")
dataset_test = load_dataset("imagefolder", data_dir="PlantVillage", split="test")    
print(f"Train dataset loaded with {len(dataset_train)} samples")
print(f"Test dataset loaded with {len(dataset_test)} samples")

# Load the feature extractor
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTImageProcessor.from_pretrained(model_id)
print("Feature extractor loaded")

# Preprocessing function
def preprocess(batch):
    images = [img.convert('RGB') if img.mode != 'RGB' else img for img in batch['image']]
    inputs = feature_extractor(images, return_tensors='pt')
    inputs['label'] = batch['label']
    return inputs

# Collate function
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

# Prepare datasets
prepared_train = dataset_train.with_transform(preprocess)
prepared_test = dataset_test.with_transform(preprocess)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load the teacher model
teacher_path = "./plantvillage_final_model"
teacher_model = ViTForImageClassification.from_pretrained(teacher_path)
teacher_model.to(device)
teacher_model.eval()
print(f"Teacher model loaded from {teacher_path}")

# Load the student model (smaller ViT variant)
student_id = 'WinKawaks/vit-tiny-patch16-224'
student_model = ViTForImageClassification.from_pretrained(
    student_id,
    num_labels=len(dataset_train.features['label'].names),
    id2label={str(i): label for i, label in enumerate(dataset_train.features['label'].names)},
    label2id={label: str(i) for i, label in enumerate(dataset_train.features['label'].names)},
    ignore_mismatched_sizes=True  # Added to handle classifier size mismatch
)
student_model.to(device)
print(f"Student model initialized from {student_id}")

# Custom Trainer for Knowledge Distillation
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha = 0.5       # Weight for distillation loss vs. hard label loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # Student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
            # Get teacher's hard predictions
            teacher_predictions = torch.argmax(teacher_logits, dim=-1)

        # Cross-entropy loss with teacher's hard predictions
        distillation_loss = torch.nn.functional.cross_entropy(student_logits, teacher_predictions)

        # Original classification loss
        classification_loss = student_outputs.loss

        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * classification_loss
        return (total_loss, student_outputs) if return_outputs else total_loss

# Training arguments for distillation
distillation_args = TrainingArguments(
    output_dir="./distilled_model_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=1,
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

# Initialize Distillation Trainer
distillation_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    model=student_model,
    args=distillation_args,
    train_dataset=prepared_train,
    eval_dataset=prepared_test,
    data_collator=collate_fn,
    processing_class=feature_extractor,  # Changed from tokenizer=feature_extractor
)

# Train the student model
print("Starting knowledge distillation...")
distillation_trainer.train()
print("Distillation training completed")

# Function to get model size in MB
def get_model_size(model, model_name="model"):
    torch.save(model.state_dict(), f"{model_name}.pt")
    size_mb = os.path.getsize(f"{model_name}.pt") / (1024 * 1024)  # Convert to MB
    os.remove(f"{model_name}.pt")
    return size_mb

# Evaluate models
def evaluate_model(model):
    trainer = Trainer(
        model=model,
        args=distillation_args,
        data_collator=collate_fn,
        eval_dataset=prepared_test,
        tokenizer=feature_extractor,
    )
    eval_results = trainer.evaluate()
    return eval_results['eval_accuracy']

# Evaluate teacher and student
print("Evaluating teacher model...")
teacher_accuracy = evaluate_model(teacher_model)
teacher_size = get_model_size(teacher_model, "teacher_model")

print("Evaluating student model...")
student_accuracy = evaluate_model(student_model)
student_size = get_model_size(student_model, "student_model")

# Save the student model
student_save_path = os.path.join(output_dir, "student_model")
student_model.save_pretrained(student_save_path)
print(f"Student model saved to {student_save_path}")

# Store results
models = ["Teacher (ViT-Base)", "Student (ViT-Tiny)"]
accuracies = [teacher_accuracy, student_accuracy]
sizes = [teacher_size, student_size]

# Print results
print("\nDistillation Results:")
print(f"{'Model':<20} {'Accuracy':<10} {'Size (MB)':<10}")
print("-" * 40)
for model_name, acc, size in zip(models, accuracies, sizes):
    print(f"{model_name:<20} {acc:<10.4f} {size:<10.2f}")

# Write results to file
with open("distillation_results.txt", "w") as f:
    f.write("Distillation Results:\n")
    f.write(f"{'Model':<20} {'Accuracy':<10} {'Size (MB)':<10}\n")
    f.write("-" * 40 + "\n")
    for model_name, acc, size in zip(models, accuracies, sizes):
        f.write(f"{model_name:<20} {acc:<10.4f} {size:<10.2f}\n")
print("Results written to distillation_results.txt")

# Plot Accuracy vs Size
plt.figure(figsize=(8, 6))
plt.scatter(sizes, accuracies, color='blue', label='Model Comparison')
for i, model_name in enumerate(models):
    plt.annotate(model_name, (sizes[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Model Size: Teacher vs Student")
plt.grid(True)
plt.legend()
plt.savefig("distillation_plot.png")
plt.show()
print("Plot saved as distillation_plot.png")