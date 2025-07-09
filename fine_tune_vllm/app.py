import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# Step 1: Load MobileNetV3 with pretrained weights
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
state_dict = MobileNet_V3_Large_Weights.DEFAULT.get_state_dict(check_hash=False)
model.load_state_dict(state_dict)

# Step 2: Replace classifier for 4-class problem
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)

# Step 3: Data transforms with normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Step 4: Load dataset and get class names
dataset = datasets.ImageFolder(root=r'D:\intern\l&t\inference\datasets', transform=transform)
class_names = dataset.classes  # Extract class names from dataset

# Step 5: Split dataset (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 6: Training setup with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train the model with validation
best_val_loss = float('inf')

for epoch in range(5):
    # Training loop
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
    
    # Save best model (with class names included)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'class_names': class_names,  # Save class names in model checkpoint
            'normalization': {'mean': [0.485, 0.456, 0.406],
                              'std': [0.229, 0.224, 0.225]}
        }, "building_class.pt")

print("Training completed. Best model saved.")
