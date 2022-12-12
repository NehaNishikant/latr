# from transformers import AutoModel, AutoConfig
from transformers import pipeline
import os

from path import path

from huggingface_hub import hf_hub_download
import torch
import json
import copy

def generate():

    """
    model = ClipCaptionModel(prefix_length)
    
    conceptual_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-conceptual-weights", filename="conceptual_weights.pt")
    model.load_state_dict(torch.load(model_path, map_location=CPU)) # captioner modeling

    model = model.to(device) 
    """

    dataset_path = path+"latr/sarcasm-dataset/"

    f_ocr = open(dataset_path+"sarcasm_ocr_train.json", "r")
    ocr_info = json.load(f_ocr)
    ocr_data = ocr_info['data']
    f_ocr.close()

    ## preprocess ##
    id2idx = {}
    for i in range(len(ocr_data)):
        id2idx[ocr_data[i]["image_id"]] = i

    f = open(dataset_path+"train.json", "r")
    info = json.load(f)
    data = info['data']
    f.close()

    device = torch.device('cuda:0')
    model = pipeline(model='nlpconnect/vit-gpt2-image-captioning', device=device)    
    
    data_augment = []
    ocr_data_augment = []
    for i in range(len(data)):
        if data[i]["answers"][0] == 'Yes':
            image_id = data[i]["image_id"]
            new_image_id = image_id+"a"

            record = copy.deepcopy(data[i])

            img_path = path+"data-of-multimodal-sarcasm-detection/dataset_image/"+image_id+".jpg"
            if not os.path.exists(img_path):
                # print("bad path: ", img_path)
                continue

            caption = model(img_path)
            record["question"] = caption[0]["generated_text"] 
            record["image_id"] = new_image_id
            record["answers"] = ["No"]
            data_augment.append(record)

            ocr_record = copy.deepcopy(ocr_data[id2idx[image_id]])
            ocr_record["image_id"] = new_image_id
            ocr_data_augment.append(ocr_record)

            # break


    f = open(dataset_path+"augmented_train.json", "w")
    d = {}
    for k,v in info.items():
        if k != "data":
            d[k] = v
    d["data"] = data_augment#+data
    json.dump(d, f, indent=4)
    f.close()

    f_ocr = open(dataset_path+"augmented_train_ocr.json", "w")
    d_ocr = {}
    for k,v in ocr_info.items():
        if k != "data":
            d_ocr[k] = v
    d_ocr["data"] = ocr_data_augment#+ocr_data
    json.dump(d_ocr, f_ocr, indent=4)
    f_ocr.close()
    


generate()
