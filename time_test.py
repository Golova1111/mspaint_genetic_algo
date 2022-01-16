import math

import numpy as np

from numba import cuda

from cuda import _cacl_elem_delta


def main():
    picture = np.empty(shape=(360, 480, 3), dtype=np.int16)
    some_image = np.empty(shape=(360, 480, 3), dtype=np.int16)
    answer = np.zeros(1, dtype=np.int64)

    d_picture = cuda.to_device(picture)
    d_some_image = cuda.to_device(some_image)
    d_answer = cuda.to_device(answer)

    # Set the number of threads in a block
    TPB = 30
    threadsperblock = (TPB, TPB, 1)

    # Calculate the number of thread blocks in the grid
    blockspergrid_x = int(math.ceil(picture.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(picture.shape[1] / threadsperblock[1]))
    blockspergrid_z = 3

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Now start the kernel
    _cacl_elem_delta[blockspergrid, threadsperblock](d_some_image, d_picture, d_answer)
    res = d_answer.copy_to_host()

    print(answer)


if __name__ == '__main__':
    main()
