import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, dataloader, class_names, device):
    """
    Evaluates a trained model on a dataloader.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for test/validation data
        class_names: List of class labels
        device: 'cuda' or 'cpu'

    Returns:
        A classification report dictionary
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy for consistency
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification Report
    print("\nClassification Report:\n")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Confusion Matrix
    conf_mat = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Also return the classification report as a dictionary (useful for saving/logging)
    return classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
