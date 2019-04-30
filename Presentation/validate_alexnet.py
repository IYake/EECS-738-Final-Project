import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
#from alexnet import alexnet


images = []
dogimg = cv2.imread('image_dog.png')##what's type of image in imagent
# birdimg = cv2.imread('image_bird.png')
# appleimg = cv2.imread('image_apple.png')
images.append(dogimg)
# images.append(birdimg)
# images.append(appleimg)

predictions = [];#use to store probabilities for each image

## apply to pretrained model
#for i in enumerate(images):
#    model = alexnet(images)
#    predictions.append(model)
predictions.append( [{'dog':0.9},{'bird':0.12},{'apple':0.05}])
# predictions.append( [{'bird':0.87},{'apple':0.33},{'bird':0.1}])
# predictions.append( [{'apple':0.88},{'bird':0.19},{'dog':0.15}])


## plot name of class and confidence
# dogimg = cv2.resize(dogimg, (1800,2000))
plt.subplot(221)
plt.imshow(cv2.cvtColor(dogimg, cv2.COLOR_BGR2RGB))
plt.title('Dog')
# plt.text(0, 2000, str(predictions[0]), size=10, rotation=0.,ha="left",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
plt.axis('off')
#
# plt.show()


#Bar Graph
x = [u'Dog', u'Cat', u'Apple', u'Bird', u'Chair', u'Lion', u'Shark', u'Table']
y = [0.04, 0.16, 0.14, 0.05, 0.15, 0.01, 0.03, 0.06]
y.sort(reverse=True)

fig, ax = plt.subplots()
width = 0.75 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Image')
plt.xlabel('x')
plt.ylabel('y')

for i, v in enumerate(y):
    plt.text(v, i, " "+str(v), color='black', va='center', fontweight='bold')

plt.show()
