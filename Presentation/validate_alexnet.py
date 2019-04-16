### skip data aug
### skip dropout
#import numpy as numpy
#import matplotlib
#import cv2#??? can we use this lib
##load all images
#
#images = []
#
##caculate imagenet mean for dataset prepocess
##mean of BGR pixel value
#img_mean = np.array([104.,117., 124.], dtype = np.float32)
##create empty Alextnet model
#model = Alextnet()
###retrive last layer and calculate softmax activation function
###softmax = e ^ (x - max(x)) / sum(e^(x - max(x))
#softmax =
##load pretrained weights into model
#model.load_initial_weights();
##iterate all images
#for img in enumerate(images):
##prepocess images, resize
#    img = cv2.resize(image.astype(np.float32), (227,227))##data augmentation
##subtract mean
#    img -= img_mean
##reshape to fit the model
#    img = img.reshape((1,227,227,3))
##calculate the class probability
##probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
##with drop out rate = 0
#probs =
##get top 3 class with highest confidence
##plot name of class and confidence
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
#from alexnet import alexnet
import matplotlib.gridspec as gridspec
from PIL import Image

## choose input images

images = []
dogimg = cv2.imread('image_dog.png')##what's type of image in imagent
birdimg = cv2.imread('image_bird.png')
appleimg = cv2.imread('image_apple.png')
images.append(dogimg)
images.append(birdimg)
images.append(appleimg)
#cv2.imshow('image',birdimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows

predictions = [];#use to store probabilities for each image

## apply to pretrained model
#for i in enumerate(images):
#    model = alexnet(images)
#    predictions.append(model)
predictions.append( [{'dog':0.9},{'bird':0.12},{'apple':0.05}])
predictions.append( [{'bird':0.87},{'apple':0.33},{'bird':0.1}])
predictions.append( [{'apple':0.88},{'bird':0.19},{'dog':0.15}])


## plot name of class and confidence
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(dogimg,str(predictions[0]), (10,500),font,1,(0,0,255), 3)
#cv2.imshow('sampe', dogimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows

dogimg = cv2.resize(dogimg, (1800,2000))
#matimagedog = mpimg.imread('image_dog.png')
plt.subplot(221)
plt.imshow(cv2.cvtColor(dogimg, cv2.COLOR_BGR2RGB))
plt.title('test title')

plt.text(0, 2000, str(predictions[0]), size=10, rotation=0.,ha="left",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
plt.axis('off')


birdimg = cv2.resize(birdimg, (1800,2000))
#matimagebird = mpimg.imread('image_bird.png')
plt.subplot(222)
plt.imshow(cv2.cvtColor(birdimg, cv2.COLOR_BGR2RGB))
plt.title('test title')
plt.text(0, 2000, str(predictions[1]), size=10, rotation=0.,ha="left",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
plt.axis('off')


appleimg = cv2.resize(appleimg,(1800,2000))
#matimageapple  = appleimg
plt.subplot(223)
plt.imshow(cv2.cvtColor(appleimg, cv2.COLOR_BGR2RGB))
plt.title('test title')
plt.text(0, 2000, str(predictions[2]), size=10, rotation=0.,ha="left",bbox=dict(boxstyle="round",ec=(0., 0., 1),fc=(1, 1, 0.9),))
plt.axis('off')
plt.show()
