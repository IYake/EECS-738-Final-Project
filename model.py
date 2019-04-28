import cnn
import numpy as np
import skimage.data
import softmax as sm
import cifar as c
import gzip
from tqdm import tqdm
import pickle

#####
#copied util functions

# def extract_data(filename, num_images, IMAGE_WIDTH):
#     print('Extracting', filename)
#     with gzip.open(filename) as bytestream:
#         bytestream.read(16)
#         buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
#         data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#         data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
#         return data
#
# def extract_labels(filename, num_images):
#     print('Extracting', filename)
#     with gzip.open(filename) as bytestream:
#         bytestream.read(8)
#         buf = bytestream.read(1 * num_images)
#         labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#     return labels

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01


def make_network(img, label, params, conv_s, pool_f, pool_s):

    [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7] = params

    #forward operations

    conv1 = cnn.conv(img, f1, b1, conv_s)
    conv1 = cnn.relu(conv1)

    pooled1 = sm.maxpool(conv1, pool_f, pool_s)
    print("pooled1", pooled1.shape)

    conv2 = cnn.conv(pooled1, f2, b2, conv_s)
    conv2 = cnn.relu(conv2)

    pooled2 = sm.maxpool(conv2, pool_f, pool_s)
    print("pooled2", pooled2.shape)

    conv3 = cnn.conv(pooled2, f3, b3, conv_s)
    conv3 = cnn.relu(conv3)

    conv4 = cnn.conv(conv3, f4, b4, conv_s)
    conv4 = cnn.relu(conv4)

    conv5 = cnn.conv(conv4, f5, b5, conv_s)
    conv5 = cnn.relu(conv5)
    print("conv5: ", conv5.shape)
    pooled3 = sm.maxpool(conv5, pool_f, pool_s)

    (nf2, dim2, _) = pooled3.shape
    print("params: ",nf2, dim2)
    fc = pooled3.reshape((nf2*dim2*dim2, 1))

    print(fc.shape, w6.shape, b6.shape)
    z = w6.dot(fc) + b6
    z = cnn.relu(z)
    out = w7.dot(z) + b7

    probs = sm.softmax(out)

    loss = cnn.cross_entropy(probs, label)

#conv1 + relu> pooled1 > conv2 + relu > pooled2 > conv3 + relu > conv4 + relu > conv5 + relu > pooled3 > fc + relu > fc
    #backward operations
    dout = probs - label

    dw7 = dout.dot(z.T) # loss gradient of final dense layer weights
    db7 = np.sum(dout, axis = 1).reshape(b7.shape) # loss gradient of final dense layer biases

    dz = w7.T.dot(dout) # loss gradient of first dense layer outputs
    dz[z<=0] = 0 # backpropagate through ReLU
    dw6 = dz.dot(fc.T)
    db6 = np.sum(dz, axis = 1).reshape(b6.shape)

    dfc = w6.T.dot(dz) # loss gradients of fully-connected layer (pooling layer)
    dpool3 = dfc.reshape(pooled3.shape) # reshape fully connected into dimensions of pooling layer

    dconv5 = cnn.pool_back(dpool3, conv5, pool_f, pool_s) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv5[conv5<=0] = 0 # backpropagate through ReLU

    dconv4, df5, db5 = cnn.conv_back(dconv5, conv4, f5, conv_s) # backpropagate previous gradient through second convolutional layer.
    dconv4[conv4<=0] = 0 # backpropagate through ReLU

    dconv3, df4, db4 = cnn.conv_back(dconv4, conv3, f4, conv_s) # backpropagate previous gradient through second convolutional layer.
    dconv3[conv3<=0] = 0 # backpropagate through ReLU

    dpool2, df3, db3 = cnn.conv_back(dconv3, conv2, f3, conv_s) # backpropagate previous gradient through second convolutional layer.
    print("conv3", conv3.shape, "dpool2: ", dpool2.shape)
    dpool2[conv3<=0] = 0 # backpropagate through ReLU

    dconv2 = cnn.pool_back(dpool2, conv2, pool_f, pool_s) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv2[conv2<=0] = 0 # backpropagate through ReLU

    dpool1, df2, db2 = cnn.conv_back(dconv2, conv1, f2, conv_s) # backpropagate previous gradient through second convolutional layer.
    dpool1[conv3<=0] = 0 # backpropagate through ReLU

    dconv1 = cnn.pool_back(dpool1, conv1, pool_f, pool_s) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv1[conv2<=0] = 0 # backpropagate through ReLU

    dimage, df1, db1 = cnn.conv_back(dconv1, img, f1, conv_s) # backpropagate previous gradient through first convolutional layer.

    grads = [df1, df2, df3, df4, df5, dw6, dw7, db1, db2, db3, db4, db5, db6, db7]

    return grads, loss

def grad_descnet(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    #update params
    [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7] = params

    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels

    cost_ = 0
    batch_size = len(batch)

    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    df3 = np.zeros(f3.shape)
    df4 = np.zeros(f4.shape)
    df5 = np.zeros(f5.shape)
    dw6 = np.zeros(w6.shape)
    dw7 = np.zeros(w7.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    db5 = np.zeros(b5.shape)
    db6 = np.zeros(b6.shape)
    db7 = np.zeros(b7.shape)

    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(f3.shape)
    v4 = np.zeros(f4.shape)
    v5 = np.zeros(f5.shape)
    v6 = np.zeros(w6.shape)
    v7 = np.zeros(w7.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    bv5 = np.zeros(b5.shape)
    bv6 = np.zeros(b6.shape)
    bv7 = np.zeros(b7.shape)

    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(f3.shape)
    s4 = np.zeros(f4.shape)
    s5 = np.zeros(f5.shape)
    s6 = np.zeros(w6.shape)
    s7 = np.zeros(w7.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    bs5 = np.zeros(b5.shape)
    bs6 = np.zeros(b6.shape)
    bs7 = np.zeros(b7.shape)

    for i in range(batch_size):

        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot

        # Collect Gradients for training example
        grads, loss = make_network(x, y, params, 1, 2, 2)
        [df1_, df2_, df3_, df4_, df5_, dw6_, dw7_, db1_, db2_, db3_, db4_, db5_, db6_, db7_] = grads

        df1+=df1_
        db1+=db1_
        df2+=df2_
        db2+=db2_
        df3+=df3_
        db3+=db3_
        df4+=df4_
        db4+=db4_
        df5+=df5_
        db5+=db5_
        dw6+=dw6_
        db6+=db6_
        dw7+=dw7_
        db7+=db7_

        cost_+= loss

    # Parameter Update

    v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam

    bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
    bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)

    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)

    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)

    v3 = beta1*v3 + (1-beta1) * df3/batch_size
    s3 = beta2*s3 + (1-beta2)*(df3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)

    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)

    v4 = beta1*v4 + (1-beta1) * df4/batch_size
    s4 = beta2*s4 + (1-beta2)*(df4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)

    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)

    v5 = beta1*v5 + (1-beta1) * df5/batch_size
    s5 = beta2*s5 + (1-beta2)*(df5/batch_size)**2
    w5 -= lr * v5 / np.sqrt(s5+1e-7)

    bv5 = beta1*bv5 + (1-beta1)*db5/batch_size
    bs5 = beta2*bs5 + (1-beta2)*(db5/batch_size)**2
    b5 -= lr * bv5 / np.sqrt(bs5+1e-7)

    v6 = beta1*v6 + (1-beta1) * dw6/batch_size
    s6 = beta2*s6 + (1-beta2)*(dw6/batch_size)**2
    w6 -= lr * v6 / np.sqrt(s6+1e-7)

    bv6 = beta1*bv6 + (1-beta1)*db6/batch_size
    bs6 = beta2*bs6 + (1-beta2)*(db6/batch_size)**2
    b6 -= lr * bv6 / np.sqrt(bs6+1e-7)

    v7 = beta1*v7 + (1-beta1) * dw7/batch_size
    s7 = beta2*s7 + (1-beta2)*(dw7/batch_size)**2
    w7 -= lr * v7 / np.sqrt(s7+1e-7)

    bv7 = beta1*bv7 + (1-beta1)*db7/batch_size
    bs7 = beta2*bs7 + (1-beta2)*(db7/batch_size)**2
    b7 -= lr * bv7 / np.sqrt(bs7+1e-7)


    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7]

    return params, cost

def train(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 2, num_filt1 = 8, num_filt2 = 8, num_filt3 = 8, num_filt4 = 8, num_filt5 = 8, batch_size = 32, num_epochs = 2, save_path = 'test.pkl'):

    # Get training data
    m =50000
    data = np.load('tiny-imagenet.npz')
    X = data['train']
    y = data['labels']
    print(X.shape, y.shape)

    y_dash = None#extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
    # X-= int(np.mean(X))
    # X/= int(np.std(X))cnm
    train_data = np.hstack((X,y))
    print(train_data.shape)

    np.random.shuffle(train_data)

    ## Initializing all the parameters
    f1, f2, f3, f4, f5, w6, w7 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (num_filt3, num_filt2, f, f), (num_filt4, num_filt3, f, f), (num_filt5, num_filt4, f, f), (128,8), (10, 128)
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    f3 = initializeFilter(f3)
    f4 = initializeFilter(f4)
    f5 = initializeFilter(f5)
    w6 = initializeWeight(w6)
    w7 = initializeWeight(w7)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((f3.shape[0],1))
    b4 = np.zeros((f4.shape[0],1))
    b5 = np.zeros((f5.shape[0],1))
    b6 = np.zeros((w6.shape[0],1))
    b7 = np.zeros((w7.shape[0],1))

    params = [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7]

    cost = []

    print("LR:"+str(lr)+", Batch Size:"+str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x,batch in enumerate(t):
            params, cost = grad_descnet(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))


    with open(save_path, 'wb') as file:
        pickle.dump(params, file)

    return cost

cost = train()














#to get extra spaces
