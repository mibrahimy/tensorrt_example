# 1. Imports & Configuration
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import os

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Loading
class MNISTDataLoader:
    def __init__(self, batch_size):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel for EfficientNet
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.batch_size = batch_size

    def get_loaders(self):
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, transform=self.transform, download=True)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, transform=self.transform, download=True)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

# 3. EfficientNet Model Definition using timm
class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetWrapper, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

# 4. Training Loop
class Trainer:
    def __init__(self, model, device, save_path='./saved_models'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'efficientnet_epoch_{epoch}.pth'))

# 5. Main Execution
def main():
    model = EfficientNetWrapper()
    data_loader = MNISTDataLoader(BATCH_SIZE)
    train_loader, test_loader = data_loader.get_loaders()
    trainer = Trainer(model, DEVICE)

    for epoch in range(EPOCHS):
        train_loss = trainer.train_epoch(train_loader)
        accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
        # Save model after each epoch
        trainer.save_model(epoch+1)

if __name__ == "__main__":
    main()
