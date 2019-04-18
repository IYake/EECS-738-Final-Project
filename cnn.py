import numpy
import sys
import pprint as pp

def conv_(img, conv_filter,stride):
    filter_size = conv_filter.shape[1]
    result = numpy.zeros((img.shape[0],img.shape[1]))
    #Looping through the image to apply the convolution operation.
    for r in numpy.uint16(numpy.arange(filter_size/stride, 
                          img.shape[0]-filter_size/stride+1)):
        for c in numpy.uint16(numpy.arange(filter_size/stride, 
                                           img.shape[1]-filter_size/stride+1)):
            """
            3D block of the image multiplied by 3D block of weights in the convolution filter
            """
            curr_region = img[r-numpy.uint16(numpy.floor(filter_size/stride)):r+numpy.uint16(numpy.ceil(filter_size/stride)), 
                              c-numpy.uint16(numpy.floor(filter_size/stride)):c+numpy.uint16(numpy.ceil(filter_size/stride)),
                              :]
            #Element-wise multiplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = numpy.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            
    #Clipping the outliers of the result matrix.
    final_result = result[numpy.uint16(filter_size/stride):result.shape[0]-numpy.uint16(filter_size/stride), 
                          numpy.uint16(filter_size/stride):result.shape[1]-numpy.uint16(filter_size/stride)]
    return final_result

def conv(img, conv_filter, stride=2):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                                img.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.

        conv_map = conv_(img, curr_filter, stride)
        feature_maps[:, :, filter_num] = conv_map[:,:] # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.
    

def pooling(feature_map, size=2, stride=2):
    #Preparing the output of the pooling operation.
    pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0]-size+1)/stride+1),
                            numpy.uint16((feature_map.shape[1]-size+1)/stride+1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in numpy.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in numpy.arange(0, feature_map.shape[1]-size+1, stride):
                pool_out[r2, c2, map_num] = numpy.max([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out

def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = numpy.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0,feature_map.shape[0]):
            for c in numpy.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
    return relu_out

#expects weights shape as (activation depth) x (volume of feature map)
def fc(feature_map,weights):
    
    if (numpy.prod(feature_map.shape) != weights.shape[-1]):
        print("Number of weights in FC doesn't match volume of feature map.")
        sys.exit()
    #Unpack feature map and return activation layer
    return numpy.dot(feature_map.reshape(-1),weights.T)
    
    
    
    
    
    