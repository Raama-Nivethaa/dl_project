# src/pipeline/predict_pipeline.py

import torch
from torchvision import transforms
from PIL import Image
import os

class Predictor:
    def __init__(self, model, class_names, device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_class = self.class_names[predicted.item()]
        return predicted_class
