import torch
import torchvision
import torchvision.transforms as transforms
import clip
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

# Load the CLIP model
model, preprocess = clip.load("RN50", device="cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Freeze all layers in the CLIP model
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer with a new linear layer
# Assuming ImageNet has 1000 classes
num_classes = 1000
ft_layer = nn.Linear(1024, num_classes).cuda()
model = model.to(device)

# Load ImageNet dataset
imagenet_train = ImageNet(root="/datasets/ilsvrc_2024-01-04_1601", split="train", transform=preprocess)
train_loader = DataLoader(imagenet_train, batch_size=64, shuffle=True)
imagenet_val = ImageNet(root="/datasets/ilsvrc_2024-01-04_1601", split="val", transform=preprocess)
val_loader = DataLoader(imagenet_val, batch_size=64, shuffle=False)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ft_layer.parameters(), lr=0.001)
num_epochs = 10

# Training loop
ft_layer.train()
for epoch in range(num_epochs):
    # add progress bar
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
    # for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        image_features = model.encode_image(images.half())
        outputs = ft_layer(image_features.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(ft_layer.state_dict(), f'finetuned_clip_rn50-{epoch}.pth')

# Evaluation loop
model.eval()
ft_layer.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        image_features = model.encode_image(images.half())
        outputs = ft_layer(image_features.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the model on the {total} test images: {100 * correct / total}%')

# Save the trained model
torch.save(ft_layer.state_dict(), 'finetuned_clip_rn50.pth')
