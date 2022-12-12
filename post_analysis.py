import json

def id2idx():

    f = open("sarcasm-dataset/sarcasm_ocr_test.json", "r")
    data = json.load(f)["data"]
    f.close()

    d = {}
    for i in range(len(data)):
        d[data[i]["image_id"]] = i

    f_out = open("sarcasm-dataset/sarcasm_ocr_test_id2idx.json", "w")
    json.dump(d, f_out)
    f_out.close()

def get_non_ocr():

    f = open("models/lightning_logs/version_577472/577472_test_wrong_list.txt", "r")
    ids = [img_id[:-1] for img_id in f.readlines()]
    f.close()

    print(ids[0])
    print("wrong: ", len(ids))

    f_ocr = open("sarcasm-dataset/sarcasm_ocr_test.json", "r")
    data = json.load(f_ocr)["data"]
    f_ocr.close() 

    f_id2idx = open("sarcasm-dataset/sarcasm_ocr_test_id2idx.json", "r")
    id2idx = json.load(f_id2idx)
    f_id2idx.close()

    counter = 0
    for img_id in ids:

        if len(data[id2idx[img_id]]["ocr_tokens"]) == 0:
            counter +=1

    print("non ocr wrong: ", counter)


def get_ocr_perf():

    f_wrong = open("models/lightning_logs/version_577472/577472_test_wrong_list.txt", "r")
    wrong_ids = [img_id[:-1] for img_id in f_wrong.readlines()]
    f_wrong.close()

    f_ocr = open("sarcasm-dataset/sarcasm_ocr_test.json", "r")
    data = json.load(f_ocr)["data"]
    f_ocr.close() 

    wrong = 0
    correct = 0

    for record in data:
        img_id = record["image_id"]
        if len(record["ocr_tokens"]) == 0:
            if img_id in wrong_ids:
                wrong +=1
                wrong_ids.remove(img_id)
            else:
                correct +=1

    
    print("correct: ", correct)
    print("wrong: ", wrong)


# print("id2idx")
# id2idx()
# print("get non ocr")
# get_non_ocr()
print("get ocr perf")
get_ocr_perf()
