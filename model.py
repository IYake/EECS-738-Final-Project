import cnn
import numpy as np
import skimage.data
import softmax as sm
import cifar as c

cifar_index = 12
labels = c.get_labels_list()
c.plot(c.img(cifar_index))

label = c.get_title(cifar_index)

img = c.img(cifar_index)
num_filters = 1
depth = img.shape[-1]
stride = 2.0
classes = 10
l1_filter = np.random.rand(num_filters,3,3,img.shape[-1])

print("\n**Working with conv layer 1**")
l1_feature_map = cnn.conv(img, l1_filter)
print("\n**ReLU**")
l1_feature_map_relu = cnn.relu(l1_feature_map)
print("\n**Pooling**")
l1_feature_map_relu_pool = cnn.pooling(l1_feature_map_relu, 2, 2)
print("\n**Fully connected**")
l1_fc_weights = np.ones((classes,np.prod(l1_feature_map_relu_pool.shape)))
l1_fc = cnn.fc(l1_feature_map_relu_pool,l1_fc_weights)

output = sm.softmax(l1_fc)
output_ex = np.zeros(output.shape)
output_ex[labels.index(label)] = 1

loss = cnn.cross_entropy(output,output_ex)
print(loss)





