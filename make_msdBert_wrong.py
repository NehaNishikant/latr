import json
from make_ocr_dataset import get_ocr_info
from make_ocr_dataset import path
from convert_to_json import convert_record

def get_wrong_list():
    
    f = open(path+"output/msdBERT_wrong.json", "r")
    img_ids = list(json.load(f).keys())
    f.close()
    return img_ids

def make_ocr_dataset():

    d_out = {
            'dataset_name': 'msdBertWrong',
            'dataset_type': 'test',
            'dataset_version': 1,
            'data': get_ocr_info(get_wrong_list())
            }
    
    f_out = open("msdBertWrong/ocr.json", "w")
    json.dump(d_out, f_out, indent=4)
    f_out.close()

def make_dataset():

    img_ids = get_wrong_list()

    f_name = path+"text/test2.txt"
    f = open(f_name, "r")

    data = []
    for line in f.readlines():

        parsed_line = eval(line)
        img_id = parsed_line[0]
        if img_id in img_ids:
            data.append(convert_record(line))
   
    f.close()

    assert(len(data) == len(img_ids))

    d_out = {
            'dataset_name': 'msdBertWrong',
            'dataset_type': 'test',
            'dataset_version': 1,
            'data': data
            }

    f_out = open("msdBertWrong/data.json", "w")
    json.dump(d_out, f_out, indent=4)
    f_out.close()

if __name__ == '__main__':

    # make_ocr_dataset()
    make_dataset()
