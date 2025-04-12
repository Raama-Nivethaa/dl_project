import torch
import torch.nn as nn
import os

# Define the same model
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 64 * 64, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# Initialize the model
model = SimpleANN()

# Ensure model folder exists
model_dir = r"C:\Users\neel\OneDrive\Desktop\dl_project\models"
os.makedirs(model_dir, exist_ok=True)

# Save model weights
model_path = os.path.join(model_dir, "my_model.pth")
torch.save(model.state_dict(), model_path)

print(f"✅ Model saved at: {model_path}")
