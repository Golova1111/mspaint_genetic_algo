import math
import time

import numpy as np
from numba import cuda


@cuda.jit
def _calc_elem_delta(curr_image, picture, answer):
    # p = cuda.grid(3)
    # answer[0] += abs(curr_image[p] - picture[p])

    # p = cuda.grid(2)
    # for i in range(3):
    #     answer[0] += abs(curr_image[p[0], p[1], i] - picture[p[0], p[1], i])
    sum = 0

    p, c = cuda.grid(2)
    for x in range(curr_image.shape[0]):
        sum += abs(picture[x, p, c] - curr_image[x, p, c])

    # answer[p, c] = sum
    answer[p + curr_image.shape[1] * c] = sum


@cuda.reduce
def sum_reduce(a, b):
    return a + b


def _calc_delta(device_pic, image):
    # answer = np.zeros((image.shape[1], 3), dtype=np.int64)
    answer = np.zeros((image.shape[1] * 3), dtype=np.int64)
    d_image = cuda.to_device(image)
    d_answer = cuda.to_device(answer)

    # Set the number of threads in a block
    TPB = 32
    # threadsperblock = (TPB, TPB, 1)
    threadsperblock = (TPB, 1)

    # Calculate the number of thread blocks in the grid
    # blockspergrid_x = int(math.ceil(image.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(image.shape[1] / threadsperblock[0]))
    blockspergrid_z = 3

    # blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    blockspergrid = (blockspergrid_y, blockspergrid_z)

    _calc_elem_delta[blockspergrid, threadsperblock](d_image, device_pic, d_answer)
    # print("inner", end - start)

    res = d_answer.copy_to_host()
    return np.sum(res)
    # return sum_reduce(d_answer)
