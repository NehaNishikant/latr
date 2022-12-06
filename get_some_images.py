import shutil

def get_some_images(fname):

    f = open(fname, "r")

    for line in f.readlines():
        img_id = line[:-1] #remove newline char

        shutil.copy("/projects/tir3/users/nnishika/MML/data-of-multimodal-sarcasm-detection/dataset_image/"+img_id+".jpg", "wrongimages")

    f.close()

get_some_images("models/lightning_logs/version_576485/wrong_list.txt")
