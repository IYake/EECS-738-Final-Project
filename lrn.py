import math
import numpy as np

def lrn(layer, radius, bias, alpha, beta):
    #based on paper, fix following values
#    radius = 5.0
#    bias = 2.0
#    alpha = 1e-4
#    beta = 0.75
    sqr_sum = np.zeros(layer.shape)
    res = np.zeros(layer.shape)
    for x in range(len(layer)):
        for y in range(len(layer[x])):
            for i in range(len(layer[x][y])):
                if (i - radius/ 2) >= 0:
                    left =  int(i - radius/ 2)
                else:
                    left = 0
                if (i + radius/2 + 1) <= layer.shape[2]:
                    right = int( i + radius/2 + 1)
                else:
                    right = layer.shape[2]
                for j in range(left, right):
                    sqr_sum[x, y, i] += layer[x, y, j] ** 2
                res[x, y , i] = float(layer[x, y, i]) /( bias + (alpha * sqr_sum[x, y, i])** beta)
    return res

if __name__ == "__main__":
    a = np.ones([3,3,3])
    c = lrn(a, 5.0, 2.0, 1e-4, 0.75)
    print c
