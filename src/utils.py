import random
import numpy as np
import pandas as pd
import statistics
import os
from tqdm import tqdm
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(inference_model, test_dataloader, index_to_target, k):
    filenames, distances, preds, top1s, modes = [], [], [], [], []
    with torch.no_grad():
        for image, filename in tqdm(test_dataloader):
            # distance & index of dataset
            distance, index = inference_model.get_nearest_neighbors(image, k=k)
            distance = distance.cpu().numpy()

            # index => target
            pred = [index_to_target[int(k)] for k in index.cpu().numpy().squeeze()]

            # top1 pred
            top1 = pred[0]

            # mode of topK
            mode = statistics.mode(pred)
            
            filenames.extend(filename)
            distances.extend(distance)
            preds.append(pred)
            top1s.append(top1)
            modes.append(mode)
        
        df = pd.DataFrame({
            "filename": filenames,
            "distance": distances,
            f"preds_top{k}": preds,
            "pred_top1": top1s,
            f"pred_mode@top{k}": modes
        })

        df_top1 = pd.DataFrame({
            "filename": filenames,
            "pred_top1": top1s,
        })

        df_mode = pd.DataFrame({
            "filename": filenames,
            f"pred_mode@top{k}": modes,
        })

    return df, df_top1, df_mode


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