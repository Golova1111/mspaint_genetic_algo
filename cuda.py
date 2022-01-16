import numpy as np
from numba import cuda


@cuda.jit
def _cacl_elem_delta(curr_image, picture, answer):
    pos = cuda.grid(3)
    answer[0] += curr_image[pos] - picture[pos]
