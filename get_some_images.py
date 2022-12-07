import shutil

from path import path

def get_some_images(fname):

    f = open(fname, "r")

    for line in f.readlines():
        img_id = line[:-1] #remove newline char

        shutil.copy(path+"data-of-multimodal-sarcasm-detection/dataset_image/"+img_id+".jpg", "wrongimages")

    f.close()

get_some_images("models/lightning_logs/version_577472/577472_test_wrong_list.txt")
