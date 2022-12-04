
print("started imports")

# import classes

from latr_finetuning import TextVQA
from latr_finetuning import DataModule 
from latr_finetuning import LaTrForVQA 
from latr_finetuning import get_data

## Default Library import

import sys
sys.path.append('src/latr')

import os
import json
import numpy as np
from tqdm.auto import tqdm
import pandas as pd

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

# checkpoint_path = "/projects/tir3/users/nnishika/MML/latr/models/lightning_logs/version_575083/epoch=0-step=2477.ckpt"
# checkpoint_path = "/projects/tir3/users/nnishika/MML/latr/models/lightning_logs/version_576209/epoch=1-step=4954.ckpt"
checkpoint_path = "/projects/tir3/users/nnishika/MML/latr/models/lightning_logs/version_576365/epoch=1-step=4954.ckpt"

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def main():

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    print(checkpoint["hyper_parameters"])
    print("epoch: ", checkpoint["epoch"])
    print(checkpoint.keys())

    model = LaTrForVQA(config, training=False)
    model.load_state_dict(checkpoint["state_dict"])

    (train_ds, val_ds, test_ds) = get_data()

    datamodule = DataModule(train_ds, val_ds, test_ds)
    trainer = pl.Trainer(logger=False, devices=1, accelerator="gpu")
    metrics = trainer.validate(model=model, dataloaders=datamodule)
    # print("metrics: ", metrics)

    """
    f = open("/projects/tir3/users/nnishika/MML/latr/models/val_metrics.json", "w")
    json.dump(metrics, f, indent=4)
    f.close()
    """

    """
    max_steps = 50000       ## 60K Steps
    latr = LaTrForVQA(config, max_steps= max_steps, pretrained_model=fpath_for_pretrained)
     
    # try:
    #     latr = latr.load_from_checkpoint(url_for_ckpt)
    #     print("Checkpoint loaded correctly")
    # except:
    #     print("Could not load checkpoint")
    #     return 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_ce_loss", mode="min"
    )
    
    # wandb.init(config=config, project="VQA with LaTr")
    # wandb_logger = WandbLogger(project="VQA with LaTr", log_model = True, entity="iakarshu")
    
    ## https://www.tutorialexample.com/implement-reproducibility-in-pytorch-lightning-pytorch-lightning-tutorial/
    pl.seed_everything(42, workers=True)
    
    trainer = pl.Trainer(
        max_steps = max_steps,
        devices=8,
        accelerator="gpu",
        # default_root_dir="logs",
#        gpus=(1 if torch.cuda.is_available() else 0),
#         accelerator="tpu",
#         devices=8,
        #logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
        default_root_dir="models/"
    )
    
    trainer.fit(latr, datamodule)
    """

if __name__ == "__main__":
    main()

"""## References:

1. [MLOps Repo](https://github.com/graviraja/MLOps-Basics) (For the integration of model and data with PyTorch Lightening) 
2. [PyTorch Lightening Docs](https://pytorch-lightning.readthedocs.io/en/stable/index.html) For all the doubts and bugs
3. [My Repo](https://github.com/uakarsh/latr) For downloading the model and pre-processing steps
4. Google for other stuffs
"""
