## Adding the path of docformer to system path
import sys
sys.path.append('src/latr/')


import os
import json
import numpy as np

import pytesseract
from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torch.nn as nn

from dataset import load_json_file, get_specific_file, resize_align_bbox, get_tokens_with_boxes, create_features
from utils import apply_mask_on_token_bbox
from modeling import LaTr_for_pretraining

PAD_TOKEN_BOX = [0, 0, 0, 0]
max_seq_len = 512
batch_size = 2
t5_model = "t5-base"

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

## Making the json entries

json_path = 'sample/OCR'
pdf_path = 'sample/pdfs'
json_entries = []
resize_shape = (1000, 1000)

for i in os.listdir(json_path):
  base_path = os.path.join(json_path, i)
  for j in os.listdir(base_path):
    json_entries.append(os.path.join(base_path, j))

## Tokenizer used in LaTr is T5Tokenizer

from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained(t5_model)

## Making the dataset object

class OCRDataset(Dataset):
  def __init__(self, json_entries, pdf_path, tokenizer,transform = None,  target_size = (1000,1000)):
    self.json_entries = json_entries
    self.pdf_path = pdf_path
    self.tokenizer = tokenizer
    self.target_size = target_size
    self.transform = transform

  def __len__(self):
    return len(self.json_entries)

  def __getitem__(self, idx):
    sample_entry = load_json_file(self.json_entries[idx])
    sample_tif_file = os.path.join(self.pdf_path, sample_entry[0].split('/')[-1])
    tif_path = get_specific_file(sample_tif_file)
    width, height = self.target_size

    ## Making the list for storing the words and coordinates

    words = []
    coordinates = []

    ## Storing the current box

    for i in sample_entry[1]['Blocks']:
      if i['BlockType']=='WORD' and i['Page']==1:
        words.append(i['Text'].lower())
        curr_box = i['Geometry']['BoundingBox']
        xmin, ymin, xmax, ymax = curr_box['Left'], curr_box['Top'], curr_box['Width']+ curr_box['Left'], curr_box['Height']+ curr_box['Top']
        curr_bbox =  resize_align_bbox([xmin, ymin, xmax, ymax], 1, 1, width, height)
        coordinates.append(curr_bbox)

    ## Similar to the docformer's create_features function, but with some changes
    img, boxes, tokenized_words = create_features(image_path = tif_path,
                                                  tokenizer = self.tokenizer,
                                                  target_size = (1000, 1000),
                                                  use_ocr = False,
                                                  bounding_box = coordinates,
                                                  words = words
                                                  )
    
    boxes = torch.as_tensor(boxes, dtype=torch.int32)
    width = (boxes[:, 2] - boxes[:, 0]).view(-1, 1)
    height = (boxes[:, 3] - boxes[:, 1]).view(-1, 1)
    boxes = torch.cat([boxes, width, height], axis = -1)

    tokenized_words = torch.as_tensor(tokenized_words, dtype=torch.int32)

    ## Applying the mask, as described in the paper, in the pre-training section
    _, masked_boxes, masked_tokenized_words = apply_mask_on_token_bbox(boxes, tokenized_words)

    if self.transform is not None:
      img = self.transform(img)
    else:
      img = transforms.ToTensor()(img)

    return  masked_boxes,masked_tokenized_words, tokenized_words

## Defining the dataset and the dataloader

ds = OCRDataset(json_entries, pdf_path, tokenizer)
dl = DataLoader(ds, shuffle = False, batch_size = batch_size)

## There can be error, if the Image file does not exist, so I request you to make changes in the dataset object, such that those address are removed

dl_entry = next(iter(dl))
masked_boxes, masked_tokenized_words, tokenized_words = dl_entry

## Transferring the tensors to appropriate device

masked_boxes = masked_boxes.to(device)
masked_tokenized_words = masked_tokenized_words.to(device)
tokenized_words = tokenized_words.to(device)

## Defining the Config file, for the model used for pre-training

config = {
    't5_model': 't5-base',
    'vocab_size': 32128,
    'hidden_state': 768,
    'max_2d_position_embeddings': 1001,
    'classes': 32128
}

pre_training_model = LaTr_for_pretraining(config, classify = True).to(device)

## Extracting the predictions from the model

extracted_feat_from_t5 = pre_training_model(masked_tokenized_words, masked_boxes)

## Initializing, the loss and optimizer

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(pre_training_model.parameters(), lr= 5e-5)

## Calculating the loss and back propagating
optimizer.zero_grad()
loss = criterion(extracted_feat_from_t5.transpose(1,2), tokenized_words.long())
loss.backward()
optimizer.step()

print("loss: ", loss)

torch.save(pre_training_model.state_dict(), "models/pretrained.pt")
