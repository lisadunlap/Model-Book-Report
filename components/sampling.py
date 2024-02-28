import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import wandb
import os
from tqdm import tqdm
import clip
import torchvision.transforms as transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
import torch
import torchvision

class Dataset(torch.utils.data.Dataset):
    def __init__(self, group1, group2, transform=None):
        self.group1 = [(g['path'], 0) for g in group1]
        self.group2 = [(g['path'], 1) for g in group2]
        self.samples = self.group1 + self.group2
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

def create_model(name="CLIP", fine_tune=False):
    if name == "CLIP":
        model, transform = clip.load("RN50", device="cuda")
    elif name == "ResNet18":
        model = torchvision.models.resnet18(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    model.fc = nn.Linear(512, 2)
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    model = model.cuda()
    return model, transform

def create_dataset(group1, group2, transform=None):
    return Dataset(group1, group2, transform=transform)

def train(model, train_loader, optimizer, epoch):
    # train for an epoch and return loss and accuracy
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    logging.info(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f} \tAccuracy: {correct / len(train_loader.dataset):.6f}')
    wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": correct / len(train_loader.dataset)})
    return train_loss, correct / len(train_loader.dataset)

def classifier_sampler(group1, group2, model="ResNet18", fine_tune=True):
    # create model and trainset
    model, transform = create_model(model, fine_tune)
    trainset = create_dataset(group1, group2, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    # train model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        train(model, train_loader, optimizer, epoch)
    model.eval()
    # save model
    torch.save(model.state_dict(), "classifier.pt")
    return get_group_confidences(trainset, model)

def get_group_confidences(dataset, model, batch_size=32):
    # Split dataset into group1 and group2
    group1_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    group2_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]

    group1_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(group1_indices))
    group2_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(group2_indices))

    def get_confidences(loader):
        model.eval()
        confidences = []
        with torch.no_grad():
            for images, _ in loader:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                confidences.extend(probs[:, 1].tolist())  # Assuming label 1 is the positive class
        return confidences

    confidences_group1 = get_confidences(group1_loader)
    confidences_group2 = get_confidences(group2_loader)
    
    return confidences_group1, confidences_group2
