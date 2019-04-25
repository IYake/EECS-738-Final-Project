import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm
import pprint as pp
import scipy
import csv
train_path = "/Users/julesgarrett/Downloads/tiny-imagenet-200/train"
label_path = "/Users/julesgarrett/Downloads/tiny-imagenet-200"
IMG_SIZE = 64


train_data =[]
for file in tqdm(os.listdir(train_path)):
    for pic in os.listdir(train_path+"/"+file+"/images"):
        img = cv2.imread(train_path+"/"+file+"/images/"+pic)
        train_data.append(np.array(img))
train_data = np.array(train_data)
# np.save('tiny-imagenet-train.npy', train_data)


# labels = read.csv(label_path+"/"+"words.txt")
labels = []
with open(label_path+"/"+'words.txt') as f:
    reader = csv.reader(f, delimiter = "\t")
    labels = list(reader)
# np.save('tiny-imagenet-train-labels.npy', labels)
np.savez_compressed('tiny-imagenet', train=train_data, labels=labels)
