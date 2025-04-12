import os
import sys
from dataclasses import dataclass
from torchvision import transforms
from src.logger.log_helper import logging
from src.exception.exception import customexception

@dataclass
class DataTransformationConfig:
    image_size: tuple = (224, 224)  # Updated for ResNet18
    mean: list = (0.485, 0.456, 0.406)  # ImageNet mean
    std: list = (0.229, 0.224, 0.225)   # ImageNet std
    augment: bool = True

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_transforms(self):
        try:
            logging.info("Creating image transformation pipeline...")

            if self.config.augment:
                train_transform = transforms.Compose([
                    transforms.Resize(self.config.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(self.config.mean, self.config.std)
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.Resize(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.config.mean, self.config.std)
                ])

            test_transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.config.mean, self.config.std)
            ])

            logging.info("Image transformations created successfully.")
            return train_transform, test_transform

        except Exception as e:
            logging.error("Error occurred in Data Transformation.")
            raise customexception(e, sys)
