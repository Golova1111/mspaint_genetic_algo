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
        alpha = rule[9]

        # -- Rectangle
        if rule[0] == 0:
            cx, cy = int((rule[1] + rule[3]) // 2), int((rule[2] + rule[4]) // 2)

            # simple "big radius check"
            r_big = (rule[1] - cx) ** 2 + (rule[2] - cy) ** 2
            if (y - cx) ** 2 + (x - cy) ** 2 > r_big:
                continue

            # get rotated coordinates
            x0 = int((rule[1] - cx) * math.cos(alpha) - (rule[2] - cy) * math.sin(alpha) + cx)
            y0 = int((rule[1] - cx) * math.sin(alpha) + (rule[2] - cy) * math.cos(alpha) + cy)

            x1 = int((rule[3] - cx) * math.cos(alpha) - (rule[4] - cy) * math.sin(alpha) + cx)
            y1 = int((rule[3] - cx) * math.sin(alpha) + (rule[4] - cy) * math.cos(alpha) + cy)

            x2 = int((rule[1] - cx) * math.cos(alpha) - (rule[4] - cy) * math.sin(alpha) + cx)
            y2 = int((rule[1] - cx) * math.sin(alpha) + (rule[4] - cy) * math.cos(alpha) + cy)

            x3 = int((rule[3] - cx) * math.cos(alpha) - (rule[2] - cy) * math.sin(alpha) + cx)
            y3 = int((rule[3] - cx) * math.sin(alpha) + (rule[2] - cy) * math.cos(alpha) + cy)

            # triangle_first:
            side_1 = (y - x1) * (y0 - y1) - (x0 - x1) * (x - y1) > 0
            side_2 = (y - x2) * (y1 - y2) - (x1 - x2) * (x - y2) > 0
            side_3 = (y - x0) * (y2 - y0) - (x2 - x0) * (x - y0) > 0

            if (side_1 == side_2 == side_3):
                curr_image[x, y, :] = rule[5], rule[6], rule[7]
                break

            # # triangle_second:
            side_1 = (y - x1) * (y3 - y1) - (x3 - x1) * (x - y1) > 0
            side_2 = (y - x0) * (y1 - y0) - (x1 - x0) * (x - y0) >= 0
            side_3 = (y - x3) * (y0 - y3) - (x0 - x3) * (x - y3) > 0

            if (side_1 == side_2 == side_3):
                curr_image[x, y, :] = rule[5], rule[6], rule[7]
                break

        # -- Triangle
        if rule[0] == 1:
            ax, ay = rule[1], rule[2]
            bx, by = rule[3], rule[4]
            cx, cy = rule[5], rule[6]

            side_1 = (y - bx) * (ay - by) - (ax - bx) * (x - by) > 0
            side_2 = (y - cx) * (by - cy) - (bx - cx) * (x - cy) > 0
            side_3 = (y - ax) * (cy - ay) - (cx - ax) * (x - ay) > 0

            if side_1 == side_2 == side_3:
                curr_image[x, y, :] = rule[7], rule[8], rule[9]
                break

        # -- Ellipse
        if rule[0] == 2:
            cx, cy = rule[1], rule[2]
            a, b = rule[3], rule[4]

            mask = (
                ((y - cx) * math.cos(alpha) + (x - cy) * math.sin(alpha)) ** 2 / (a * a) +
                ((y - cx) * math.sin(alpha) - (x - cy) * math.cos(alpha)) ** 2 / (b * b)
                < 1
            )
            if mask:
                curr_image[x, y, :] = rule[5], rule[6], rule[7]
                break


def _gen_picture(picture):
    lparts = len(picture.parts)
    decoded_parts = np.zeros((lparts, 10))

    for i, part in enumerate(picture.parts):
        decoded_parts[lparts - 1 - i] = part._get_repr()

    d_image = cuda.to_device(picture.picture)
    d_parts = cuda.to_device(decoded_parts)

    # Set the number of threads in a block
    TPB = 20
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

    # ============ instant score calculation
    # =======================================

    answer = np.zeros((picture.picture.shape[1] * 3), dtype=np.int64)
    d_answer = cuda.to_device(answer)

    # Calculate the number of thread blocks in the grid
    blockspergrid_y = int(math.ceil(picture.picture.shape[1] / threadsperblock[0]))
    blockspergrid_z = 3
    threadsperblock = (TPB, 1)
    blockspergrid = (blockspergrid_y, blockspergrid_z)

    _calc_elem_delta[blockspergrid, threadsperblock](picture.d_picture, d_image, d_answer)
    res = d_answer.copy_to_host()
    picture._score = np.sum(res)

    return picture
