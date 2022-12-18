import pandas as pd
import os
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig

from src.utils import set_seed



@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # read data frame
    train_master = pd.read_csv(cfg.train_master_dir)

    train_num_classes = train_master['flag'].nunique()
    assert (
        cfg.num_classes == train_num_classes
    ), f'num_classes should be {train_num_classes}'

    # image name list
    image_name_list = train_master['file_name'].values

    # label list
    label_list = train_master['flag'].values

    # split train & val
    x_train, x_val, y_train, y_val = train_test_split(
        image_name_list,
        label_list,
        test_size=cfg.train_test_split.test_size,
        stratify=label_list,
        random_state=cfg.train_test_split.random_state
    )

    # set seed
    set_seed(cfg.seed)

    # device

    # directory to save models
    SAVE_MODEL_PATH = cfg.save_model_dir
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)