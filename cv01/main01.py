import argparse
import os
import sys
from collections import defaultdict

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from wandb_config import WANDB_PROJECT, WANDB_ENTITY
import matplotlib.pyplot as plt


class DenseNet(nn.Module):
    def __init__(self, dropout=0.2):
        super(DenseNet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # obrázek musíme nejprve "rozbalit" do 1D vektoru uděláme to ale až od první dimenze, protože první dimenze je batch a
        # tu chceme zachovat
        x = x.flatten(1)

        # poté postupně aplikujeme lineární vrstvy, dropout a aktivační funkce
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)

        output = F.log_softmax(x, dim=1)

        return output


class Net(nn.Module):
    def __init__(self, dropout=0.2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader, config):
    SAMPLES = 200
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, d in enumerate(test_loader):

            data, target = d
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += output.size()[0]
            if i * config["batch_size"] == SAMPLES:
                break
    test_loss /= total  ## delit totalem ..
    acc = 100. * correct / total
    print(f"test: avg-loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.0f}%)\n")
    return test_loss, acc


def count_norm(model_params):
    total_norm = 0
    for p in model_params:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        else:
            print("NO GRAD")
            pass
    total_norm = total_norm ** 0.5
    return total_norm


def main(config: dict):
    # ukažme rozdíl cpu a gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv01"], config=config)

    EPOCHS = 2
    BATCH_SIZE = config["batch_size"]
    LOG_INTERVAL = 1

    LR = config["lr"]

    config["use_normalization"] = False

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    
    DATA_ROOT = "/storage/plzen1/home/vladar21/nlp-cv01/data"


    dataset1 = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)

    if config["model"] == "dense":
        model = DenseNet(dropout=config.get("dp", 0.2)).to(device)
    else:
        model = Net(dropout=config.get("dp", 0.2)).to(device)
    wandb.watch(model, log="all")
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR)
        
    if config["scheduler"] == "exponential":
        lr_scheduler = ExponentialLR(optimizer, gamma=config.get("gamma", 0.9))
    elif config["scheduler"] == "step":
        lr_scheduler = StepLR(optimizer, step_size=config.get("step_size", 5), gamma=config.get("gamma", 0.9))

    # training loop
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            #pass
            #plt.imshow(data[0].permute(1, 2, 0))
            #plt.imshow(data[0].permute(2, 1, 0))
            #print(data)
            #plt.show()
            #print(target)

            #print(data[0])
            #print(data[0].shape)


            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)  
            output = model(data)  # forward                       

            norm = count_norm(model.parameters())  

            loss = F.nll_loss(output, target)  

            loss.backward()  
            optimizer.step()  

            print(f"e{epoch} b{batch_idx} s{batch_idx * BATCH_SIZE}]\t"  
                  f"Loss: {loss.item():.6f}")  

            my_log = {"train_loss": loss.item()}   
            my_log["lr"] = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else config['lr']  

            if batch_idx % LOG_INTERVAL == LOG_INTERVAL - 1:  
                model.eval()  
                test_loss, test_acc = test(model, device, test_loader, config)  
                model.train()  

                my_log["test_loss"] = test_loss
                my_log["test_acc"] = test_acc
                wandb.log(my_log)

        if lr_scheduler is not None:
            lr_scheduler.step()            



if __name__ == '__main__':
    config = defaultdict(lambda: False)

    print(config)
    # add parameters lr,optimizer,dp
    main(config)

