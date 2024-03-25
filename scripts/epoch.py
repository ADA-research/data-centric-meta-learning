import time
from resnet import ResNet
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from contextlib import nullcontext

def epoch(
        model_obj: ResNet, 
        device,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        val: bool,
        n: int,
        verbose = True
    ):
    t0 = time.time()
    model_obj.eval() if val else model_obj.train()

    running_loss = 0.
    running_corrects = 0
    with torch.no_grad() if val else nullcontext():
        for (inputs, labels) in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if not val: optimizer.zero_grad() 
            outputs = model_obj(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if not val:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    loss = (running_loss / len(dataloader.sampler))
    acc = (running_corrects / len(dataloader.sampler)) * 100.
    if verbose: print(f'[{"val" if val else "train"} #{n}] Loss: {loss:.4f} Acc: {acc:.4f}% Time: {time.time() - t0:.4f}s')
    return loss, acc
