import math
from numpy import array
import numpy as np
<<<<<<< HEAD


def softmax(fully_connected_8):
    out = np.exp(fully_connected_8) # exponentiate vector of raw predictions
    return out/np.sum(out)



def maxpool(image, f=2, s=2):
    n_c, h_prev, w_prev = image.shape

    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1

    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((n_c, h, w))


=======
'''implementation for softmax function
    apply the standard exponential function to each element of the input, a list of results of dividing by sum of those exponentials is returned.'''
def softmax(fully_connected_8):
    # w_exp = []
    # b = max(fully_connected_8) #because exponential normalization is shift invariant
    # for i in fully_connected_8:
    #     w_exp.append(math.exp(i-b))
    # sum_w_exp = sum(w_exp)
    # softmax = []
    # for i in w_exp:
    #     softmax.append(i / sum_w_exp)
    # for i in softmax:
    #     print('{0:6f}'.format(i)) #round values to 6 places after decimal
    # return array(softmax)
    out = np.exp(fully_connected_8) # exponentiate vector of raw predictions
    return out/np.sum(out)



def maxpool(image, f=2, s=2):
    '''
    Downsample input `image` using a kernel size of `f` and a stride of `s`
    '''
    n_c, h_prev, w_prev = image.shape

    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1

    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((n_c, h, w))


>>>>>>> 77121589df99b8233288c646a8c568117476b377
    # slide the window over every part of the image using stride s. Take the maximum value at each step.
    for i in range(n_c):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + f <= w_prev:
                # choose the maximum value within the window at each step and store it to the output matrix
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled
