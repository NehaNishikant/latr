import json
from make_ocr_dataset import get_ocr_info
from make_ocr_dataset import path
from convert_to_json import convert_record

def get_wrong_list(dataset_name):
    
    f = open(path+"output/"+dataset_name+"_wrong.json", "r")
    img_ids = list(json.load(f).keys())
    f.close()
    return img_ids

def make_ocr_dataset(dataset_name):

    d_out = {
            'dataset_name': dataset_name+'Wrong',
            'dataset_type': 'test',
            'dataset_version': 1,
            'data': get_ocr_info(get_wrong_list(dataset_name))
            }
    
    print(dataset_name+"Wrong/ocr.json")
    f_out = open(dataset_name+"Wrong/ocr.json", "w")
    json.dump(d_out, f_out, indent=4)
    f_out.close()

def make_dataset(dataset_name):

    img_ids = get_wrong_list(dataset_name)

    f_name = path+"text/test2.txt"
    f = open(f_name, "r")

    data = []
    counter = 0
    wronglen = len(img_ids)
    for line in f.readlines():

        parsed_line = eval(line)
        img_id = parsed_line[0]
        if img_id in img_ids:
            data.append(convert_record(line))
            img_ids.remove(img_id)

            counter +=1

            if counter % 100 == 0:
                print(counter)
   
    f.close()

    assert(len(data) == wronglen)

    d_out = {
            'dataset_name': dataset_name+'Wrong',
            'dataset_type': 'test',
            'dataset_version': 1,
            'data': data
            }

    print(dataset_name+"Wrong/data.json")
    f_out = open(dataset_name+"Wrong/data.json", "w")
    json.dump(d_out, f_out, indent=4)
    f_out.close()

if __name__ == '__main__':

    make_ocr_dataset("msdBert")
    # make_dataset("msdBert")
    make_ocr_dataset("headache")
    # make_dataset("headache")
