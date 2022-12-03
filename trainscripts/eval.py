
print("started imports")

# import classes

# from latr_finetuning import TextVQA
from latr_finetuning import DataModule 
# from latr_finetuning import LaTrForVQA 

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
max_seq_len = 10 #512
batch_size = 1
target_size = (500,384) ## Note that, ViT would make it 224x224 so :(
t5_model = "t5-base"

"""
## Appending the ocr and json path
import os
base_path = 'sarcasm-dataset/'
train_ocr_json_path = os.path.join(base_path, 'sarcasm_ocr_train.json')
train_json_path = os.path.join(base_path, 'train.json')

val_ocr_json_path = os.path.join(base_path, 'sarcasm_ocr_valid.json')
val_json_path = os.path.join(base_path, 'valid.json')

## Loading the files

train_ocr_json = json.load(open(train_ocr_json_path))['data']
train_json = json.load(open(train_json_path))['data']

val_ocr_json = json.load(open(val_ocr_json_path))['data']
val_json = json.load(open(val_json_path))['data']

## Useful for the key-value extraction

train_json_df = pd.DataFrame(train_json)
train_ocr_json_df = pd.DataFrame(train_ocr_json)

val_json_df = pd.DataFrame(val_json)
val_ocr_json_df = pd.DataFrame(val_ocr_json)

## Converting list to string

train_json_df['answers'] = train_json_df['answers'].apply(lambda x: " ".join(list(map(str, x))))
val_json_df['answers']   = val_json_df['answers'].apply(lambda x: " ".join(list(map(str, x))))

## Dropping of the images which doesn't exist, might take some time

# base_img_path = os.path.join('./', 'train_images')
# print("base image path: ", base_img_path)
base_img_path = "/projects/tir3/users/nnishika/MML/data-of-multimodal-sarcasm-detection/dataset_image"
train_json_df['path_exists'] = train_json_df['image_id'].progress_apply(lambda x: os.path.exists(os.path.join(base_img_path, x)+'.jpg'))
train_json_df = train_json_df[train_json_df['path_exists']==True]

val_json_df['path_exists'] = val_json_df['image_id'].progress_apply(lambda x: os.path.exists(os.path.join(base_img_path, x)+'.jpg'))
val_json_df = val_json_df[val_json_df['path_exists']==True]

## Dropping the unused columns

# train_json_df.drop(columns = ['flickr_original_url', 'flickr_300k_url','image_classes', 'question_tokens', 'path_exists'
#                               ], axis = 1, inplace = True)
# val_json_df.drop(columns = ['flickr_original_url', 'flickr_300k_url','image_classes', 'question_tokens', 'path_exists'
#                               ], axis = 1, inplace = True)

## Deleting the json

del train_json
del train_ocr_json
del val_json
del val_ocr_json

## Grouping for the purpose of feature extraction
grouped_df = train_json_df.groupby('image_id')

## Getting all the unique keys of the group by object
keys = list(grouped_df.groups.keys())

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

train_ds = TextVQA(base_img_path = base_img_path,
                   json_df = train_json_df,
                   ocr_json_df = train_ocr_json_df,
                   tokenizer = tokenizer,
                   transform = None, 
                   max_seq_length = max_seq_len, 
                   target_size = target_size
                   )


val_ds = TextVQA(base_img_path = base_img_path,
                   json_df = val_json_df,
                   ocr_json_df = val_ocr_json_df,
                   tokenizer = tokenizer,
                   transform = None, 
                   max_seq_length = max_seq_len, 
                   target_size = target_size
                   )

idx = 0

train_ds.json_df

encoding = train_ds[idx]  ## Might take time as per the processing
for key in list(encoding.keys()):
  print_statement = '{0: <30}'.format(str(key) + " has a shape:")
  print(print_statement, encoding[key].shape)

## Sample Img, Sample box, sample words, sample answer, sample question

s_img = encoding['img']
s_boxes = encoding['boxes']
s_words = encoding['tokenized_words']
s_ans = encoding['answer']
s_ques = encoding['question']
"""


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
    'seq_len': 10
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = "/projects/tir3/users/nnishika/MML/latr/models/epoch=0-step=2477.ckpt"

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def main():

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage).eval()
    print(checkpoint["hyper_parameters"])
    print("epoch: ", checkpoint["epoch"])
    print(checkpoint.keys())

    datamodule = DataModule(train_ds, val_ds)
    trainer = Trainer()
    trainer.test(model, dataloaders=datamodule)

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
