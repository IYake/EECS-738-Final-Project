import math
import numpy as np
def lrn(layer, radius, alpha, beta, name, bias):
    #based on paper, fix following values
    radius = 5.0
    bias = 2.0
    alpha = 1e-4
    beta = 0.75

    sqr_sum = layer
    res = layer
    for x in layer:
        for y in layer[x]:
            for i in layer[x][y]:
                sqr_sum[x, y, i] = sum(layer[x, y, max(0, i - radius / 2): min(95, i + radius / 2)] ** 2)
                res[x, y , i] = layer[x, y, i] / (bias + alpha * sqr_sum[x, y, i])** beta
    return res    