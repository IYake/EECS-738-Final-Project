import math

'''implementation for softmax function
    apply the standard exponential function to each element of the input, a list of results of dividing by sum of those exponentials is returned.'''
def softmax(fully_connected_8):
    w_exp = []
    for i in fully_connected_8:
        w_exp.append(math.exp(i))
    sum_w_exp = sum(w_exp)
    softmax = []
    for i in w_exp:
        softmax.append(i / sum_w_exp)
    for i in softmax:
        print('{0:6f}'.format(i)) #round values to 6 places after decimal
    return softmax

