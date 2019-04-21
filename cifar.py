import numpy as np
from numpy import linalg as LA
import pickle
import re
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import sys

def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1
 
pd_tr = pd.DataFrame()
tr_y = pd.DataFrame()
 
for i in range(1,6):
    data = unpickle('cifar-10-batches-py/data_batch_' + str(i))
    pd_tr = pd_tr.append(pd.DataFrame(data[b'data']))
    tr_y = tr_y.append(pd.DataFrame(data[b'labels']))
    pd_tr['labels'] = tr_y
 
tr_x = np.asarray(pd_tr.iloc[:, :3072])
tr_y = np.asarray(pd_tr['labels'])
ts_x = np.asarray(unpickle('cifar-10-batches-py/test_batch')[b'data'])
ts_y = np.asarray(unpickle('cifar-10-batches-py/test_batch')[b'labels'])    
labels = unpickle('cifar-10-batches-py/batches.meta')[b'label_names']
 
def plot_CIFAR(ind):
    arr = tr_x[ind]
    R = arr[0:1024].reshape(32,32)
    G = arr[1024:2048].reshape(32,32)
    B = arr[2048:].reshape(32,32)
 
    img = np.dstack((R,G,B))
    title = re.sub('[!@#$b]', '', str(labels[tr_y[ind]]))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(img,interpolation='bicubic')
    ax.set_title('Category = '+ title,fontsize =15)

def resize(ind,h,w):
    arr = tr_x[ind]
    R = arr[0:1024].reshape(32,32)
    G = arr[1024:2048].reshape(32,32)
    B = arr[2048:].reshape(32,32)
    img = np.dstack((R,G,B))    
    res = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
    R = res[:,:,0]
    G = res[:,:,1]
    B = res[:,:,2]
    R = R.reshape(h*w)
    G = G.reshape(h*w)
    B = B.reshape(h*w)
    img = np.concatenate((R,G,B))
    return img

def plot(img):
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(img,interpolation='bicubic')

def closest(ind,search_size):
    plot_CIFAR(ind)
    img = tr_x[ind]
    img = np.transpose(np.reshape(img,(3, 32,32)), (1,2,0))
    minIdx = 0
    minDif = LA.norm(abs(tr_x[ind]-tr_x[0]),2)
    for i in range(100):
        dif = LA.norm(abs(tr_x[ind]-tr_x[i]),2)
        if (dif < minDif and i != ind):
            minDif = dif
            minIdx = i
    plot_CIFAR(minIdx)
    
def closestK(ind, search_size, K = 1):
    smallest = []
    for i in range(K):
        smallest.append((i,dif(ind,i)))
    
    for i in range(search_size):
        if (i != ind):
            replacement = max(smallest, key=lambda x:x[1])
            if dif(ind,i) < replacement[1]:
                smallest[smallest.index(replacement)] = (i,dif(ind,i))
    
    plot_CIFAR(ind)
    for tup in smallest:
        plot_CIFAR(tup[0])

def dif(idx1,idx2):
    return LA.norm(abs(tr_x[idx1]-tr_x[idx2]),2)

def plot_row(img,h,w):
    img = twoToThree(img,h,w)
    plot(img)
    

def twoToThree(row,h,w):
    R = row[0:h*w].reshape(h,w)
    G = row[h*w:2*h*w].reshape(h,w)
    B = row[2*h*w:].reshape(h,w)
    img = np.dstack((R,G,B))
    return img

def row(ind):
    return tr_x[ind]

def img(ind):
    return twoToThree(row(ind),32,32)

def get_title(ind):
    title = re.sub('[!@#$b]', '', str(labels[tr_y[ind]]))
    return title

def get_labels_list():
    labels = []
    for i in range(10):
        labels.append(get_title(i))
    return labels

