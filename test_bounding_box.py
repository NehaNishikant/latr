import pytesseract
from pytesseract import Output
from PIL import Image
import json


def make_dataset(split):

    if split == "train":
        f_in = ?
    elif split == "test":
        f_in = ?
    else:
        f_in = ?
    
    data = []

    for _ in _:

        img = Image.open('testimages/726506255683039232.jpg')
        d = pytesseract.image_to_data(img, output_type=Output.DICT)

        n_boxes = len(d['level'])
        ocr_tokens = []
        ocr_info = []
        for i in range(n_boxes):
            
            token = d['text'][i].strip()

            if d['conf'][i] < 0 or len(token) == 0:
                continue

            print(token, d['conf'][i])

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
                'image_id': ?,
                'ocr_tokens': ocr_tokens,
                'ocr_info': ocr_info
            })

    d_out = {
        'dataset_name': 'sarcasm',
        'dataset_type': 'train',
        'dataset_verion': 1,
        'data': data
        }
    json.dump(d_out, f_out, indent=4)
            