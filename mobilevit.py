import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# Simplified MobileViT-S model (replace with full impl. from paper)
class FireDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.classifier = nn.Linear(192, 2)  # 2 classes: fire/non-fire

    def forward(self, x):
        return self.classifier(self.backbone(x))

# Train
model = FireDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in train_loader:  # Create your DataLoader
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
