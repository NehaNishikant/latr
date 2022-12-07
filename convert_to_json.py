import pandas as pd
import json

from path import path


def convert_record(line):

    line = line.rstrip()[2:-1]
    #print(line)
    
    double_quote = False
    id_index = line.find("', '")
    if id_index < 0:
        id_index = line.index("', \"")
        double_quote = True
    image_id = line[:id_index]
    #print(image_id)

    line = line[id_index+4:]
    #print(line)

    comment_index = line.index('", ') if double_quote else line.index("', ")
    comment = line[:comment_index]
    #print(comment)

    line = line[comment_index+3:]
    #print(line)

    label = line[-1]
    #print(label)

    answer = 'Yes' if label == '1' else '0'

    data = {
        "question": comment,
        'image_id': image_id,
        'image_width': 1024,
        'image_height': 1024,
        'answers': [answer]
    }

    return data


def convert(dataset_split):

    dataset_split_in = dataset_split
    if dataset_split != "train":
        dataset_split_in += "2"
    fname = path+'data-of-multimodal-sarcasm-detection/' + dataset_split_in + '.txt'
    # df = pd.read_csv(fname)

    # print(df.head(1))

    dataset = {
        "dataset_type": dataset_split,
        "dataset_name": "headacheboy",
        "dataset_version": "1",
        "data": []
    }

    with open(fname) as file:
        for line in file:
            dataset['data'].append(convert_record(line))


    with open('sarcasm-dataset/' + dataset_split + '.json', 'w') as f:
        json.dump(dataset, f)

# convert("train")
# convert("valid")
# convert("test")
