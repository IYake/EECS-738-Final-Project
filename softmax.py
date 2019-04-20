import math

'''implementation for softmax function
    apply the standard exponential function to each element of the input, a list of results of dividing by sum of those exponentials is returned.'''
def softmax(fully_connected_8):
    w_exp = []
    for i in fully_connected_8:
        w_exp.append(math.exp(i))
        print i
    sum_w_exp = sum(w_exp)
    softmax = []
    for i in w_exp:
        softmax.append(round(i / sum_w_exp, 6)) #set precision to 6
    print softmax
    return softmax

