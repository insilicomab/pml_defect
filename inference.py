"""
The inference results are saved to a csv file.
Usage:
    Inference with model on wandb:
        python inference.py \
        --model_name {model name storaged in wandb} \
        --wandb_run_path {wandb_run_path} \
        --image_size {image size default: 224}
        --embedding_size {embedder output size default: 512} \
        --k {top@k default: 10}
        
"""
import pandas as pd

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_metric_learning
from pytorch_metric_learning.utils.inference import InferenceModel
import wandb

from src.dataset import DefectDataset, TestTransforms
from src.model import get_model
from src.utils import predict


def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/pml_defect/input/train.csv')
    test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/pml_defect/input/sample_submission.csv', header=None)

    # train image name list & label list
    image_name_list = train['id']
    label_list = train['target']

    # index2target: key=index, value=target
    index2target = train['target'].to_dict()

    # test image name list & dummy label list
    x_test = test[0].values
    dummy = test[0].values

    # dataset
    dataset = DefectDataset(
        image_name_list,
        label_list,
        img_dir='/content/drive/MyDrive/Colab Notebooks/pml_defect/input/train_data',
        transform=TestTransforms(image_size=args.image_size),
        phase='test'
    )

    # test dataset
    test_dataset = DefectDataset(
        x_test,
        dummy,
        img_dir='/content/drive/MyDrive/Colab Notebooks/defect/input/test_data',
        transform=TestTransforms(image_size=args.image_size),
        phase='test'
    )

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    model = get_model(
        model_name=args.timm_name,
        pretrained=False,
        embedding_size=args.embedding_size,
    )

    # restore model in wandb
    best_model = wandb.restore(f'{args.model_name}.pth', run_path=args.wandb_run_path)

    model.load_state_dict(torch.load(best_model.name, map_location=torch.device(device)))
    model.eval()
    model.to(device)

    # inference
    im = InferenceModel(model)
    im.train_knn(dataset)

    df, df_top1, df_mode = predict(
        inference_model=im,
        test_dataloader=test_dataloader,
        index_to_target=index2target,
        k=args.k
    )

    df.to_csv(f'/content/drive/MyDrive/Colab Notebooks/pml_defect/submit/inference_{args.model_name}.csv', sep=',', index=None)
    df_top1.to_csv(f'/content/drive/MyDrive/Colab Notebooks/pml_defect/submit/submission_top1_{args.model_name}.csv', sep=',', header=None, index=None)
    df_mode.to_csv(f'/content/drive/MyDrive/Colab Notebooks/pml_defect/submit/submission_mode_{args.model_name}.csv', sep=',', header=None, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--timm_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--wandb_run_path', type=str)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    
    args = parser.parse_args()

    main(args)