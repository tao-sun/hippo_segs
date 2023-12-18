from statistics import mean
from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset

from torchmetrics.classification import Dice

from model import SNN
from data import HippoDataset


available_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")  # torch.device("mps")

def get_confusion_matrix(target, prediction):
    labels = np.unique(target)
    confusion_mat = confusion_matrix(target.flatten(),
                                     prediction.flatten(),
                                     labels=labels)
    return confusion_mat


def dice_ratio(target, prediction):
    confusion_mat = get_confusion_matrix(target, prediction)
    true_positives = np.float64(confusion_mat.diagonal())
    # print(f"true_positives: {true_positives.shape}")

    class_actual_sums = np.sum(confusion_mat, axis=1)
    false_negative = class_actual_sums - true_positives

    class_predicted_sums = np.sum(confusion_mat, axis=0)
    false_positive = class_predicted_sums - true_positives

    dice_ratio = (2*true_positives) / (2*true_positives + false_negative + false_positive)
    return dice_ratio



learning_rate = 0.01
dataset_path = "./data/data3"
batch_size = 3
epochs = 500


train_subjects = range(1, 100)
test_subjects = range(100, 111)
frames = 48
train_set = HippoDataset(dataset_path, train_subjects, frames)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    # num_workers=1, 
    shuffle=True)
test_set = HippoDataset(dataset_path, test_subjects, frames)
test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        # num_workers=1, 
        shuffle=False)


ce_criterion = nn.CrossEntropyLoss()

net  = SNN()
net.to(available_device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
dice = Dice(num_classes=2, ignore_index=0).to(available_device)
dice_values = []
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

for epoch in range(epochs):
    # print(f"epoch: {epoch}")
    # Randomly shuffle train
    epoch_loss = 0
    # if (epoch + 1) % 400 == 0:
    #     optimizer.param_groups[0]['lr'] = learning_rate / 10

    # for x, y in train_loader:
    pbar = tqdm(train_loader, desc=f"epoch: {epoch}")
    for x, y in pbar:
        # print(f"step: {i}")
        # zero the parameter gradients
        x = x.to(available_device)
        y = y.to(available_device)
        optimizer.zero_grad()

        # forward + backward + optimize
        logits = net(x)
        # print(f"logits shape: {logits.dtype}, y shape: {y.dtype}")
        loss = ce_criterion(logits, y)
        
        pbar.set_postfix({'loss': loss.item()})
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    scheduler.step(epoch_loss)
    
    if epoch >= 10:
        pbar = tqdm(test_loader, desc=f"testing: ")
        for x, y in pbar:
            x = x.to(available_device)
            y = y.to(available_device)

            logits = net(x)
            loss = ce_criterion(logits, y)
            pbar.set_postfix({'loss': loss.item()})
            
            labels = torch.argmax(logits, dim=1)
            # print(f"labels shape: {labels.shape}, y shape: {y.shape}")
            
            # dice_value = dice_ratio(y.detach().cpu().numpy(), labels.detach().cpu().numpy())
            # pbar.set_postfix({'dice': dice_value[1]})
            # dice_values.append(dice_value[1])

            dice_value = dice(labels, y)
            pbar.set_postfix({'dice': dice_value.item()})
            dice_values.append(dice_value.item())
            
        # avg_dice = torch.mean(torch.stack(dice_values))
        avg_dice = mean(dice_values)
        print(f"dice: {avg_dice}")