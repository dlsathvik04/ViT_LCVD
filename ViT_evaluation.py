from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load model and dataset
model_path = 'plantvillage_final_model_tiny'
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
test_dataset = load_dataset("imagefolder", "PlantVillage", split="test")

# Prediction function
def get_predictions(model, dataset):
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for sample in dataset:
            inputs = processor(sample['image'], return_tensors="pt")
            logits = model(**inputs).logits
            pred_label = logits.argmax(-1).item()
            
            true_labels.append(sample['label'])
            pred_labels.append(pred_label)
    
    return true_labels, pred_labels

# Get predictions
true_labels, pred_labels = get_predictions(model, test_dataset)
label_names = test_dataset.features['label'].names

# Print classification report
print(classification_report(true_labels, pred_labels, target_names=label_names))

# Plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
ConfusionMatrixDisplay(cm, display_labels=label_names).plot(cmap='Blues', xticks_rotation='vertical')
plt.show()