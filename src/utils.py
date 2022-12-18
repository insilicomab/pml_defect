import random
import numpy as np
import os
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelCheckpoint():
    def __init__(self, save_model_path, filename):
        self.best_score=np.inf
        self.save_path_filename = os.path.join(save_model_path, f'{filename}.pth')
    
    def __call__(self, model, current_score):
        if current_score < self.best_score:
            self.best_score = current_score
            torch.save(model.state_dict(), self.save_path_filename)


class EarlyStopping:
    def __init__(self, patience=10, verbose=1):
        self.epoch = 0
        self.pre_loss = float('inf')
        self.patience = patience
        self.verbose = verbose
        
    def __call__(self, current_loss):
        if self.pre_loss < current_loss:
            self.epoch += 1
            if self.epoch > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self.epoch = 0
            self.pre_loss = current_loss
        return False