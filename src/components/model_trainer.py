import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataclasses import dataclass
from src.logger.log_helper import logging
from src.exception.exception import customexception
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import models

@dataclass
class ModelTrainerConfig:
    model_save_path: str = os.path.join("artifacts", "alzheimer_resnet_model.pth")
    num_epochs: int = 10
    learning_rate: float = 0.0001
    num_classes: int = 4

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False  # Freeze feature extractor

        # Replace final FC layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.config.num_classes)
        )
        return model.to(self.device)

    def train_model(self, train_loader, test_loader, class_names):
        try:
            logging.info("Model training started.")
            model = self.build_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.fc.parameters(), lr=self.config.learning_rate)

            for epoch in range(self.config.num_epochs):
                model.train()
                running_loss = 0
                for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                logging.info(f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}")

            torch.save(model.state_dict(), self.config.model_save_path)
            logging.info(f"Model saved at {self.config.model_save_path}")

            self.evaluate_model(model, test_loader, class_names)
            return model

        except Exception as e:
            logging.error("Error during model training.")
            raise customexception(e, sys)

    def evaluate_model(self, model, test_loader, class_names):
        logging.info("Evaluating the model...")
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, target_names=class_names))

        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
