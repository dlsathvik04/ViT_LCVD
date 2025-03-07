from datasets import load_dataset
import torch
from transformers import ViTImageProcessor, TrainingArguments, ViTForImageClassification, Trainer
import os
import numpy as np
from torch.utils.data import WeightedRandomSampler

# model id to be used
model_id = 'google/vit-base-patch16-224-in21k'
print(f"Model ID selected: {model_id}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# device
print(f"Using device: {device}")

dataset_train = load_dataset("imagefolder", "PlantVillage", split="train")  # streaming=True for lazy loading

# dataset_train, dataset_train.features
print(f"Training dataset loaded with {len(dataset_train)} samples")

dataset_test = load_dataset("imagefolder", "PlantVillage", split="test")

# dataset_test
print(f"Test dataset loaded with {len(dataset_test)} samples")

# dataset_train[0]
print("Sample 0 inspected from training dataset")

feature_extractor = ViTImageProcessor.from_pretrained(
    model_id,
    ignore_mismatched_sizes=True
)
# feature_extractor
print(f"Feature extractor loaded from {model_id}")

def preprocess(batch):
    # take a list of PIL images and turn them to pixel values
    images = [img.convert('RGB') if img.mode != 'RGB' else img for img in batch['image']]
    inputs = feature_extractor(
        images,
        return_tensors='pt'
    )
    # include the labels
    inputs['label'] = batch['label']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

prepared_train = dataset_train.with_transform(preprocess)
prepared_test = dataset_test.with_transform(preprocess)

labels = dataset_train.features['label'].names

model = ViTForImageClassification.from_pretrained(
    model_id,
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: str(i) for i, label in enumerate(labels)}
)
model.to(device)

training_args = TrainingArguments(
    output_dir="./plantvillage_model",
    per_device_train_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=3,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=prepared_train,
    eval_dataset=prepared_test,
    tokenizer=feature_extractor,
)

# Initial training
if os.path.exists("./plantvillage_checkpoints"):
    # Get the latest checkpoint
    checkpoints = [dir for dir in os.listdir("./plantvillage_checkpoints") if dir.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join("./plantvillage_checkpoints", latest_checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        train_results = trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("No checkpoint found. Starting training from scratch.")
        train_results = trainer.train()
else:
    print("No checkpoint directory found. Starting training from scratch.")
    train_results = trainer.train()

# Save and log initial training results
trainer.save_model("./plantvillage_final_model")
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
print("Initial training completed and model saved")

# Balanced retraining
print("Starting balanced retraining...")

# Calculate class weights for balanced sampling
class_counts = np.bincount(dataset_train['label'])
num_classes = len(class_counts)
class_weights = 1.0 / class_counts
sample_weights = torch.DoubleTensor([class_weights[label] for label in dataset_train['label']])

# Create sampler for balanced retraining
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset_train),
    replacement=True
)

# Update training arguments for retraining
balanced_training_args = TrainingArguments(
    output_dir="./plantvillage_model_balanced",
    per_device_train_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=2,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)

# Create new trainer with balanced sampler
balanced_trainer = Trainer(
    model=model,
    args=balanced_training_args,
    data_collator=collate_fn,
    train_dataset=prepared_train,
    eval_dataset=prepared_test,
    tokenizer=feature_extractor,
    train_sampler=sampler
)

# Perform balanced retraining
balanced_results = balanced_trainer.train()

# Save and log balanced retraining results
balanced_trainer.save_model("./plantvillage_final_model_balanced")
balanced_trainer.log_metrics("train", balanced_results.metrics)
balanced_trainer.save_metrics("train", balanced_results.metrics)
balanced_trainer.save_state()
print("Balanced retraining completed and model saved")