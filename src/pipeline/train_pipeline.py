# src/pipeline/train_pipeline.py

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import evaluate_model
import torch

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Data transformation
    transformation = DataTransformation()
    train_loader, test_loader = transformation.initialize_data_transformation(train_path, test_path)

    # Model training
    trainer = ModelTrainer()
    model = trainer.initate_model_training(train_loader, test_loader, device)

    # Evaluation
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    evaluate_model(model, test_loader, class_names, device)
