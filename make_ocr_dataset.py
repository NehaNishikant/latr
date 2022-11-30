import pytesseract
from pytesseract import Output
from PIL import Image
import json
import argparse


def make_dataset(split):

    path = "/projects/tir3/users/nnishika/MML/data-of-multimodal-sarcasm-detection/"

    textpath = path+"text/"
    if split == "train":
        f_in = open(textpath+"train.txt", "r")
    else:
        f_in = open(textpath+split+"2.txt", "r")
    
    data = []
    for line in f_in.readlines():

        img_id = eval(line)[0]

        imgpath = path+"dataset_image/"+img_id+'.jpg'
        try:
            img = Image.open(imgpath)
        except:
            print("bad img id: ", img_id)
            continue

        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['level'])
        ocr_tokens = []
        ocr_info = []
        for i in range(n_boxes):
            
            token = d['text'][i].strip()

            if d['conf'][i] < 0 or len(token) == 0:
                continue

            # print(token, d['conf'][i])

            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

            ocr_tokens.append(token)
            ocr_info.append({
                "word": token,
                "bounding_box": {
                    "top_left_x": x,
                    "top_left_y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0 #an assumption, but this is all pytesseract gives us
                }
            })

        data.append({
            'image_id': img_id,
            'ocr_tokens': ocr_tokens,
            'ocr_info': ocr_info
        })

    d_out = {
        'dataset_name': 'sarcasm',
        'dataset_type': split,
        'dataset_verion': 1,
        'data': data
        }
    f_out = open("dataset/sarcasm_ocr_"+split+".json", "w")
    json.dump(d_out, f_out, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=str, default=None)
    args = parser.parse_args()
    make_dataset(args.split)
