import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def save_model(model, path):
    """
    Save the PyTorch model to disk.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    """
    Load the PyTorch model from disk.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def set_seed(seed=42):
    """
    Ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_image(image_path):
    """
    Load and preprocess an image for model inference.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def visualize_predictions(images, labels, preds, class_names):
    """
    Show sample predictions (for classification tasks).
    """
    plt.figure(figsize=(12, 8))
    for i in range(min(len(images), 8)):
        image = images[i].permute(1, 2, 0).numpy()
        image = (image * 0.5) + 0.5  # Unnormalize
        plt.subplot(2, 4, i + 1)
        plt.imshow(image)
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
