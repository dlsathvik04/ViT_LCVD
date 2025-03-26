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
        self.temperature = 2.0  # Temperature for softening probabilities
        self.alpha = 0.5       # Weight for distillation loss vs. hard label loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # Softmax with temperature
        student_soft = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)

        # Distillation loss (KL divergence)
        distillation_loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

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
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=1,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
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
    processing_class=feature_extractor,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(1) == p.label_ids).mean()}
)

# Train the student model
print("Starting knowledge distillation...")
distillation_trainer.train()
print("Distillation training completed")

results_dir = "./results_distillation"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    

def plot_and_save_metrics(trainer, model_name, phase="initial"):
    log_history = trainer.state.log_history
    train_steps, train_loss = [], []
    val_steps, val_loss = [], []
    val_acc_steps, val_acc = [], []
    
    # Extract steps and corresponding metrics
    for log in log_history:
        if 'step' in log:
            step = log['step']
            if 'loss' in log:
                train_steps.append(step)
                train_loss.append(log['loss'])
            if 'eval_loss' in log:
                val_steps.append(step)
                val_loss.append(log['eval_loss'])
            if 'eval_accuracy' in log:
                val_acc_steps.append(step)
                val_acc.append(log['eval_accuracy'])
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label='Training Loss')
    plt.plot(val_steps, val_loss, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves ({phase})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_loss_{phase}.png'))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(val_acc_steps, val_acc, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy Curve ({phase})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_accuracy_{phase}.png'))
    plt.close()


plot_and_save_metrics(distillation_trainer, model_name="soft distillation")