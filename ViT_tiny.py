from datasets import load_dataset
import torch
from transformers import ViTImageProcessor, TrainingArguments, ViTForImageClassification, Trainer
import os
import numpy as np
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt

# Create results directory
results_dir = "./results_tiny"
os.makedirs(results_dir, exist_ok=True)

# Model id to be used
model_id = 'google/vit-tiny-patch16-224'
print(f"Model ID selected: {model_id}")

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# Load predefined train and test splits
train_dataset = load_dataset("imagefolder", data_dir="PlantVillage", split="train")
test_dataset = load_dataset("imagefolder", data_dir="PlantVillage", split="test")

# Split train into train and validation (80/20 of train)
train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

print(f"Training dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")


# Feature extractor
feature_extractor = ViTImageProcessor.from_pretrained(model_id, ignore_mismatched_sizes=True)
print(f"Feature extractor loaded from {model_id}")

def preprocess(batch):
    images = [img.convert('RGB') if img.mode != 'RGB' else img for img in batch['image']]
    inputs = feature_extractor(images, return_tensors='pt')
    inputs['label'] = batch['label']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

# Prepare datasets
prepared_train = train_dataset.with_transform(preprocess)
prepared_val = val_dataset.with_transform(preprocess)
prepared_test = test_dataset.with_transform(preprocess)

labels = train_dataset.features['label'].names

# Initialize model
model = ViTForImageClassification.from_pretrained(
    model_id,
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: str(i) for i, label in enumerate(labels)}
)
model.to(device)

# Function to plot and save metrics
def plot_and_save_metrics(trainer, model_name, phase="initial"):
    log_history = trainer.state.log_history
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for log in log_history:
        if 'loss' in log:
            train_loss.append(log['loss'])
        if 'eval_loss' in log:
            val_loss.append(log['eval_loss'])
        if 'eval_accuracy' in log:
            val_acc.append(log['eval_accuracy'])
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves ({phase})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_loss_{phase}.png'))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy Curve ({phase})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_accuracy_{phase}.png'))
    plt.close()

# Initial training setup
training_args = TrainingArguments(
    output_dir="./plantvillage_model_tiny",
    per_device_train_batch_size=32,  # Increased batch size for tiny model
    evaluation_strategy="steps",
    num_train_epochs=3,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=prepared_train,
    eval_dataset=prepared_val,
    tokenizer=feature_extractor,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(1) == p.label_ids).mean()}
)

# Initial training
checkpoint_dir = "./plantvillage_checkpoints_tiny"
if os.path.exists(checkpoint_dir):
    checkpoints = [dir for dir in os.listdir(checkpoint_dir) if dir.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        train_results = trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("No checkpoint found. Starting training from scratch.")
        train_results = trainer.train()
else:
    print("No checkpoint directory found. Starting training from scratch.")
    train_results = trainer.train()

# Save model and plot metrics
trainer.save_model("./plantvillage_final_model_tiny")
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
plot_and_save_metrics(trainer, model_id.split('/')[-1], "initial")
print("Initial training completed and model saved")

# Balanced retraining
print("Starting balanced retraining...")
class_counts = np.bincount(train_dataset['label'])
class_weights = 1.0 / class_counts
sample_weights = torch.DoubleTensor([class_weights[label] for label in train_dataset['label']])

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)

balanced_training_args = TrainingArguments(
    output_dir="./plantvillage_model_tiny_balanced",
    per_device_train_batch_size=32,  # Increased batch size for tiny model
    evaluation_strategy="steps",
    num_train_epochs=2,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

balanced_trainer = Trainer(
    model=model,
    args=balanced_training_args,
    data_collator=collate_fn,
    train_dataset=prepared_train,
    eval_dataset=prepared_val,
    tokenizer=feature_extractor,
    train_sampler=sampler,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(1) == p.label_ids).mean()}
)

balanced_results = balanced_trainer.train()

# Save model and plot metrics
balanced_trainer.save_model("./plantvillage_final_model_tiny_balanced")
balanced_trainer.log_metrics("train", balanced_results.metrics)
balanced_trainer.save_metrics("train", balanced_results.metrics)
balanced_trainer.save_state()
plot_and_save_metrics(balanced_trainer, model_id.split('/')[-1], "balanced")
print("Balanced retraining completed and model saved")

# Evaluate on test set
test_results = balanced_trainer.evaluate(prepared_test)
print("Test set evaluation results:", test_results)