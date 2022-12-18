from tqdm import tqdm
import numpy as np
import torch
import pytorch_metric_learning
from pytorch_metric_learning import testers
import wandb


def train(model, loss_fn, train_dataloader, device, epoch, optimizer, loss_optimizer):
    model.train()
    total_loss = 0
    for data, labels in tqdm(train_dataloader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        loss_optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_fn(embeddings, labels)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()
        total_loss += loss.detach().cpu().numpy()
    print(f'Epoch {epoch}: train_loss = {loss}')
    return total_loss


def test(model, loss_fn, test_dataloader, device, epoch):
    model.eval()
    with torch.no_grad():    
        total_loss = 0
        for data, labels in tqdm(test_dataloader):
            data, labels = data.to(device), labels.to(device)
            embeddings = model(data)
            loss = loss_fn(embeddings, labels)
            total_loss += loss.detach().cpu().numpy()
        print(f'Epoch {epoch}: test_loss = {loss}')
        return total_loss


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def calculate_accuracy(train_dataset, test_dataset, model, accuracy_calculator):
    model.eval()
    train_embeddings, train_labels = get_all_embeddings(train_dataset, model)
    test_embeddings, test_labels = get_all_embeddings(test_dataset, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print('Computing accuracy')
    print(f'test_Precision@1 = {accuracies["precision_at_1"]:.03f}, test_NMI = {accuracies["NMI"]:.03f}, test_AMI = {accuracies["AMI"]:.03f}, test_r_precision = {accuracies["r_precision"]:.03f}, test_mean_average_precision_at_r = {accuracies["mean_average_precision_at_r"]:.03f}')
    return accuracies


def train_model(
    cfg, 
    model, 
    loss, 
    train_dataset, 
    test_dataset, 
    train_dataloader, 
    test_dataloader, 
    device, 
    accuracy_calculator, 
    optimizer, 
    loss_optimizer=None, 
    model_checkpoint=None, 
    early_stopping=None
    ):
    best_loss = np.inf
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train(model, loss, train_dataloader, device, epoch, optimizer, loss_optimizer)
        val_loss = test(model, loss, test_dataloader, device, epoch)
        accuracies=calculate_accuracy(train_dataset, test_dataset, model, accuracy_calculator)
        if model_checkpoint:
            model_checkpoint(model, val_loss)
        if early_stopping:
            if early_stopping(val_loss):
                break


def train_model_wandb(
    cfg, 
    model, 
    loss, 
    train_dataset, 
    test_dataset, 
    train_dataloader, 
    test_dataloader, 
    device, 
    accuracy_calculator, 
    optimizer, 
    loss_optimizer=None, 
    model_checkpoint=None, 
    early_stopping=None
    ):
    best_loss = np.inf
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train(model, loss, train_dataloader, device, epoch, optimizer, loss_optimizer)
        val_loss = test(model, loss, test_dataloader, device, epoch)
        accuracies=calculate_accuracy(train_dataset, test_dataset, model, accuracy_calculator)
        wandb.log({'train_loss':train_loss, 'val_loss':val_loss})
        wandb.log({
            'test_Precision@1':accuracies["precision_at_1"], 
            'test_NMI':accuracies["NMI"], 
            'test_AMI':accuracies["AMI"], 
            'test_r_precision':accuracies["r_precision"], 
            'test_mean_average_precision_at_r':accuracies["mean_average_precision_at_r"]
        })
        if model_checkpoint:
            model_checkpoint(model, val_loss)
        if early_stopping:
            if early_stopping(val_loss):
                break