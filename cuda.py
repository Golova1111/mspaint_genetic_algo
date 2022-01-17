import math
import time

import numpy as np
from numba import cuda


@cuda.jit
def _calc_elem_delta(curr_image, picture, answer):
    sum = 0

    p, c = cuda.grid(2)
    for x in range(curr_image.shape[0]):
        sum += abs(picture[x, p, c] - curr_image[x, p, c])

    # answer[p, c] = sum
    answer[p + curr_image.shape[1] * c] = sum


def _calc_delta(device_pic, image):
    # answer = np.zeros((image.shape[1], 3), dtype=np.int64)
    answer = np.zeros((image.shape[1] * 3), dtype=np.int64)
    d_image = cuda.to_device(image)
    d_answer = cuda.to_device(answer)

    # Set the number of threads in a block
    TPB = 30
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


# =================================
# =================================
# =================================


@cuda.jit
def _gen_elem_picture(curr_image, picture_rules):
    x, y = cuda.grid(2)

    for rule in picture_rules:
        if rule[0] == 0:
            if (rule[1] <= x < rule[3]) and (rule[2] <= y < rule[4]):
                curr_image[x, y, :] = rule[5], rule[6], rule[7]
        if rule[0] == 1:
            ax, ay = rule[1], rule[2]
            bx, by = rule[3], rule[4]
            cx, cy = rule[5], rule[6]

            side_1 = (y - bx) * (ay - by) - (ax - bx) * (x - by) > 0
            side_2 = (y - cx) * (by - cy) - (bx - cx) * (x - cy) > 0
            side_3 = (y - ax) * (cy - ay) - (cx - ax) * (x - ay) > 0

            if side_1 == side_2 == side_3:
                curr_image[x, y, :] = rule[7], rule[8], rule[9]
        if rule[0] == 2:
            cx, cy = rule[1], rule[2]
            a, b = rule[3], rule[4]

            mask = (y - cx) ** 2 / (a * a) + (x - cy) ** 2 / (b * b) < 1
            if mask:
                curr_image[x, y, :] = rule[5], rule[6], rule[7]


def _gen_picture(picture):
    decoded_parts = np.zeros((len(picture.parts), 10))

    for i, part in enumerate(picture.parts):
        decoded_parts[i] = part._get_repr()

    d_image = cuda.to_device(picture.picture)
    d_parts = cuda.to_device(decoded_parts)

    # Set the number of threads in a block
    TPB = 30
    # threadsperblock = (TPB, TPB, 1)
    threadsperblock = (TPB, TPB)

    # Calculate the number of thread blocks in the grid
    blockspergrid_x = int(math.ceil(picture.w / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(picture.h / threadsperblock[1]))

    # blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _gen_elem_picture[blockspergrid, threadsperblock](d_image, d_parts)
    # print("inner", end - start)

    new_image = d_image.copy_to_host()
    picture.picture = new_image
    return picture
