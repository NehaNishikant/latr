
print("started imports")

# import classes

from latr_finetuning import TextVQA
from latr_finetuning import DataModule 
from latr_finetuning import LaTrForVQA 
from latr_finetuning import get_data
from latr_finetuning import path

## Default Library import

import sys
sys.path.append('src/latr')

import os
import json
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import argparse

## For the purpose of displaying the progress of map function
tqdm.pandas()

## Visualization libraries
import pytesseract
from PIL import Image, ImageDraw

## Specific libraries of LaTr
import torch
import torch.nn as nn

## Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

## Warm-up
import pytorch_warmup as warmup

import pytorch_lightning as pl

print("finished imports")

## Setting the hyperparameters as well as primary configurations

PAD_TOKEN_BOX = [0, 0, 0, 0]
batch_size = 1
from utils import max_seq_len
target_size = (500,384) ## Note that, ViT would make it 224x224 so :(
t5_model = "t5-base"


## Appending the ocr and json path
import os


## Tokenizer import

from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained(t5_model)


# sent = tokenizer.encode("what is the url in the picture?", max_length = 512, truncation = True, padding = 'max_length', return_tensors = 'pt')[0]
# dec_sent = tokenizer.decode(sent, skip_special_tokens = True)
# dec_sent

## Defining the pytorch dataset

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms


"""## 6. Modeling Part 🏎️
1. Firstly, we would define the pytorch model with our configurations
2. Secondly, we would encode it in the PyTorch Lightening module, and boom 💥 our work of defining the model is done
"""


from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

path += "latr/models/lightning_logs/"
def main():

    print(args.model_path)
    
    ## keys are img, boxes, tokenized_words, answer, question

    config = {
        't5_model': 't5-base',
        'vocab_size': 32128,
        'hidden_state': 768,
        'max_2d_position_embeddings': 1001,
        'classes': 32128,
        'seq_len': max_seq_len
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # checkpoint_path = path+"version_575083/epoch=0-step=2477.ckpt"
    # checkpoint_path = path+"version_576209/epoch=1-step=4954.ckpt"
    # checkpoint_path = path+"version_576365/epoch=1-step=4954.ckpt"
    # checkpoint_path = path+"version_576485/epoch=1-step=4954.ckpt"
    checkpoint_path = path+args.model_path #"version_576605/epoch=1-step=4954-v1.ckpt"

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    print(checkpoint["hyper_parameters"])
    print("epoch: ", checkpoint["epoch"])
    print(checkpoint.keys())

    model = LaTrForVQA(config, training=False, prefix=args.model_path[8:14])
    model.load_state_dict(checkpoint["state_dict"])

    print("model loaded")

    # this is for evaluating msd bert
    kwargs = { #only val matters
            'base_path': 'headacheWrong', #msdBertWrong/',
            'train_fname': 'data.json',
            'val_fname': 'data.json',
            'test_fname': 'data.json',
            'train_ocr_fname': 'ocr.json',
            'val_ocr_fname': 'ocr.json',
            'test_ocr_fname': 'ocr.json'
            }
    (train_ds, val_ds, test_ds) = get_data(ablation=args.ablation) #, **kwargs)

    print("data loaded")

    datamodule = DataModule(train_ds, val_ds, test_ds)
    trainer = pl.Trainer(logger=False, devices=1, accelerator="gpu")

    print("trainer initialized")
    
    if args.split == "val":
        metrics = trainer.validate(model=model, dataloaders=datamodule)
    else:
        print("Test")
        metrics = trainer.test(model=model, dataloaders=datamodule)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--split', type=str, default="val")
    args = parser.parse_args()

    main()

"""## References:

1. [MLOps Repo](https://github.com/graviraja/MLOps-Basics) (For the integration of model and data with PyTorch Lightening) 
2. [PyTorch Lightening Docs](https://pytorch-lightning.readthedocs.io/en/stable/index.html) For all the doubts and bugs
3. [My Repo](https://github.com/uakarsh/latr) For downloading the model and pre-processing steps
4. Google for other stuffs
"""
