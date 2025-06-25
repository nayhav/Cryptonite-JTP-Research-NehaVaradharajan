import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the CNN architecture
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Flatten(),      
            nn.Dropout(0.3),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer, scheduler
model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training
num_epochs = 10
train_losses = []

print("Training starts\n")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
print("\nTraining complete.")

# Evaluation
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"\n Test Accuracy: {accuracy:.2f}%")
import os
os.makedirs("results", exist_ok=True)
# Visualize some predictions
import random

classes = train_dataset.classes
model.eval()

# Get one batch from test_loader
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Randomly select 8 images
indices = random.sample(range(len(images)), 8)
sample_images = images[indices]
sample_labels = labels[indices]

with torch.no_grad():
    outputs = model(sample_images.to(device))
    _, preds = torch.max(outputs, 1)

# Plot
plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    img = sample_images[i].squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {classes[sample_labels[i]]}\nPred: {classes[preds[i]]}")
    plt.axis('off')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/sample_predictions.png")
print("✅ Sample predictions saved to results/sample_predictions.png")
plt.show()


# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=train_dataset.classes, 
            yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()

# Classification report
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

import os
os.makedirs("results", exist_ok=True)

print("train_losses:", train_losses)  # 🧪 Check list contents

plt.figure(figsize=(8, 5))
plt.plot(train_losses, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()

plt.savefig("results/loss_curve.png")
print("🖼️  Loss curve saved to results/loss_curve.png")

plt.show()


# Save model
torch.save(model.state_dict(), "fashion_cnn.pth")
