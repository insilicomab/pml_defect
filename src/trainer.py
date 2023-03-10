from tqdm import tqdm
import numpy as np
import torch
import pytorch_metric_learning
from pytorch_metric_learning import testers
import wandb


class Trainer():
    def __init__(
        self,
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
        loss_optimizer=None
        ):
        self.cfg=cfg
        self.model=model
        self.loss=loss
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader
        self.device=device
        self.accuracy_calculator=accuracy_calculator
        self.optimizer=optimizer
        self.loss_optimizer=loss_optimizer

    
    def _train(self, model, loss_fn, train_dataloader, device, epoch, optimizer, loss_optimizer):
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
            total_loss += loss.detach().cpu().numpy().item() * data.size(0)
        train_loss = total_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch}: train_loss = {train_loss}')
        return train_loss


    def _test(self, model, loss_fn, test_dataloader, device, epoch):
        model.eval()
        with torch.no_grad():    
            total_loss = 0
            for data, labels in tqdm(test_dataloader):
                data, labels = data.to(device), labels.to(device)
                embeddings = model(data)
                loss = loss_fn(embeddings, labels)
                total_loss += loss.detach().cpu().numpy().item() * data.size(0)
            test_loss = total_loss / len(test_dataloader.dataset)
            print(f'Epoch {epoch}: test_loss = {test_loss}')
            return test_loss
    

    def _get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester()
        return tester.get_all_embeddings(dataset, model)
    

    def _calculate_accuracy(self, train_dataset, test_dataset, model, accuracy_calculator):
        model.eval()
        train_embeddings, train_labels = self._get_all_embeddings(train_dataset, model)
        test_embeddings, test_labels = self._get_all_embeddings(test_dataset, model)
        train_labels = train_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
        accuracies = accuracy_calculator.get_accuracy(
            test_embeddings, train_embeddings, test_labels, train_labels, False
        )
        print('Computing accuracy')
        print(f'test_Precision@1 = {accuracies["precision_at_1"]:.03f}, test_NMI = {accuracies["NMI"]:.03f}, test_AMI = {accuracies["AMI"]:.03f}, test_r_precision = {accuracies["r_precision"]:.03f}, test_mean_average_precision_at_r = {accuracies["mean_average_precision_at_r"]:.03f}')
        return accuracies
    

    def train_model(self, model_checkpoint=None, early_stopping=None):
        best_loss = np.inf
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train(self.model, self.loss, self.train_dataloader, self.device, epoch, self.optimizer, self.loss_optimizer)
            val_loss = self._test(self.model, self.loss, self.test_dataloader, self.device, epoch)
            accuracies=self._calculate_accuracy(self.train_dataset, self.test_dataset, self.model, self.accuracy_calculator)
            if model_checkpoint:
                model_checkpoint(self.model, val_loss)
            if early_stopping:
                if early_stopping(val_loss):
                    break
    

    def train_model_wandb(self, model_checkpoint=None, early_stopping=None):
        best_loss = np.inf
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train(self.model, self.loss, self.train_dataloader, self.device, epoch, self.optimizer, self.loss_optimizer)
            val_loss = self._test(self.model, self.loss, self.test_dataloader, self.device, epoch)
            accuracies=self._calculate_accuracy(self.train_dataset, self.test_dataset, self.model, self.accuracy_calculator)
            wandb.log({'train_loss':train_loss, 'val_loss':val_loss})
            wandb.log({
                'test_Precision@1':accuracies["precision_at_1"], 
                'test_NMI':accuracies["NMI"], 
                'test_AMI':accuracies["AMI"], 
                'test_r_precision':accuracies["r_precision"], 
                'test_mean_average_precision_at_r':accuracies["mean_average_precision_at_r"]
            })
            if model_checkpoint:
                model_checkpoint(self.model, val_loss)
            if early_stopping:
                if early_stopping(val_loss):
                    break