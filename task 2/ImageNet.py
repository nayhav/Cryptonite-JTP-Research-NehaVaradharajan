import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

MODEL_NAME = "resnet18"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_FOLDER = "./my_images"            # Inference folder
LABELS_PATH = "imagenet_classes.txt"

DO_FINE_TUNE = True                 # Training dataset
CUSTOM_DATASET_PATH = "./custom_dataset"
NUM_CUSTOM_CLASSES = 3
EPOCHS = 20
BATCH_SIZE = 16
CSV_OUTPUT_PATH = "predictions.csv"
TOP_K = min(5, NUM_CUSTOM_CLASSES if DO_FINE_TUNE else 1000)

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
if not os.path.exists(LABELS_PATH):
    urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)
with open(LABELS_PATH) as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# Load base pretrained model
model = models.__dict__[MODEL_NAME](pretrained=True)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last residual block (layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CUSTOM_CLASSES if DO_FINE_TUNE else 1000)
)
model.to(DEVICE)

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# TRAINING
if DO_FINE_TUNE:
    from torchvision.datasets import ImageFolder
    print(" Fine-tuning on custom dataset")
    full_dataset = ImageFolder(root=CUSTOM_DATASET_PATH, transform=transform)

    # Split: 80% training, 20% validation
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []
    all_preds = []
    all_labels = []

    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0.0
        val_loss = 0.0

        #  Training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation 
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        model.train()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    print(" Fine-tuning completes here")


# INFERENCE
class InferenceDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path


print("️ Images found for inference:", os.listdir(IMAGE_FOLDER))

dataset = InferenceDataset(IMAGE_FOLDER, transform)
loader = DataLoader(dataset, batch_size=1)

# Write CSV
with open(CSV_OUTPUT_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Rank", "Label", "Probability (%)"])

    model.eval()
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_idxs = probs.topk(TOP_K, dim=1)

            for i in range(images.size(0)):
                image_path = paths[i]
                image_name = os.path.basename(image_path)
                print(f"\n Image: {image_name}")

                top_labels = []
                top_scores = []

                for rank in range(TOP_K):
                    label_idx = top_idxs[i][rank].item()

                    
                    label = (imagenet_labels if not DO_FINE_TUNE else ImageFolder(CUSTOM_DATASET_PATH).classes)[label_idx]

                    prob = top_probs[i][rank].item() * 100
                    print(f"  🔹 {label:30} — {prob:.2f}%")
                    writer.writerow([image_name, rank + 1, label, f"{prob:.2f}"])
                    top_labels.append(label)
                    top_scores.append(prob)

                # Plot predictions
                def plot_predictions(image_path, labels, probs):
                    img = Image.open(image_path).convert("RGB")
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title("Input Image")

                    plt.figure(figsize=(8, 4))
                    plt.barh(labels, probs, color="skyblue")
                    plt.xlabel("Probability")
                    plt.title("Top Predictions")
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.show()

                plot_predictions(image_path, top_labels, top_scores)

print(f"\n Prediction results saved to `{CSV_OUTPUT_PATH}`")
