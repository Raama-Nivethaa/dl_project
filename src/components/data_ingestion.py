import os
import sys
from dataclasses import dataclass
from src.logger.log_helper import logging
from src.exception.exception import customexception
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import evaluate_model

@dataclass
class DataIngestionConfig:
    train_dir: str = os.path.join("C:\\Users\\neel\\OneDrive\\Desktop\\dl_project\\data_split\\train", "train")
    test_dir: str = os.path.join("C:\\Users\\neel\\OneDrive\\Desktop\\dl_project\\data_split\\test", "test")
    batch_size: int = 32
    image_size: tuple = (64, 64)

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def _get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return train_transform, test_transform

    def initiate_data_ingestion(self):
        logging.info("Starting image data ingestion process...")
        try:
            train_transform, test_transform = self._get_transforms()

            train_dataset = datasets.ImageFolder(root=self.config.train_dir, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=self.config.test_dir, transform=test_transform)

            logging.info(f"Train classes found: {train_dataset.classes}")

            # Handle class imbalance
            targets = [label for _, label in train_dataset]
            class_counts = Counter(targets)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in targets]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=sampler)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

            logging.info("Data ingestion completed successfully.")

            return train_loader, test_loader, train_dataset.classes

        except Exception as e:
            logging.error("Exception occurred in Data Ingestion stage.")
            raise customexception(e, sys)
        
        transformer = DataTransformation()
        train_transform, test_transform = transformer.get_transforms()
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
        
        trainer = ModelTrainer()
        
        model = trainer.train_model(train_loader, test_loader, class_names=["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"])
        evaluate_model(model, test_loader, class_names=['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'], device=device)