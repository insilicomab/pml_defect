import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pytorch_metric_learning
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import hydra
from omegaconf import DictConfig
import wandb

from src.dataset import DefectDataset, Transforms
from src.model import get_model, get_arcfaceloss, get_optimizer
from src.utils import set_seed, ModelCheckpoint, EarlyStopping
from src.trainer import Trainer


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # read data frame
    train_master = pd.read_csv(cfg.train_master_dir)

    train_num_classes = train_master['target'].nunique()
    assert (
        cfg.num_classes == train_num_classes
    ), f'num_classes should be {train_num_classes}'

    # image name list
    image_name_list = train_master['id'].values

    # label list
    label_list = train_master['target'].values

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

    # initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config={
            'data': os.path.basename(cfg.wandb.data_dir),
            'model': cfg.wandb.model_name,
        }
    )

    # save hydra's config in wandb cloud storage
    shutil.copy2('config/config.yaml', os.path.join(wandb.run.dir, 'hydra_config.yaml'))

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # directory to save models
    SAVE_MODEL_PATH = cfg.save_model_dir
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # dataset
    train_dataset = DefectDataset(
        image_name_list=x_train,
        label_list=y_train,
        img_dir=cfg.img_dir,
        transform=Transforms(cfg=cfg),
        phase='train'
    )
    val_dataset = DefectDataset(
        image_name_list=x_val,
        label_list=y_val,
        img_dir=cfg.img_dir,
        transform=Transforms(cfg=cfg),
        phase='val'
    )

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=cfg.train_dataloader.shuffle,
        num_workers=cfg.train_dataloader.num_workers,
        pin_memory=cfg.train_dataloader.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=cfg.val_dataloader.shuffle,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=cfg.val_dataloader.pin_memory,
    )

    # model
    model = get_model(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        embedding_size=cfg.embedding_size,
    )
    model = model.to(device)

    # loss
    loss, _, _ = get_arcfaceloss(cfg=cfg)
    loss = loss.to(device)

    # optimizer
    optimizer = get_optimizer(cfg, model)
    loss_optimizer = get_optimizer(cfg, loss)

    # accuracy calculator
    accuracy_calculator = AccuracyCalculator(k=cfg.accuracy_calculator.k)

    # model checkpoint
    model_checkpoint = ModelCheckpoint(
        save_model_path=wandb.run.dir,
        filename=wandb.run.name
    )

    # early stopping
    early_stopping = EarlyStopping(
        patience=cfg.earlystopping.patience,
        verbose=cfg.earlystopping.verbose
    )

    # wandb
    wandb.watch(model, log="all")

    # train
    trainer = Trainer(
        cfg=cfg, 
        model=model, 
        loss=loss, 
        train_dataset=train_dataset, 
        test_dataset=val_dataset, 
        train_dataloader=train_dataloader, 
        test_dataloader=val_dataloader, 
        device=device, 
        accuracy_calculator=accuracy_calculator, 
        optimizer=optimizer, 
        loss_optimizer=loss_optimizer,
    )

    trainer.train_model_wandb(
        model_checkpoint=model_checkpoint,
        early_stopping=early_stopping,
    )


if __name__ == "__main__":
    main()