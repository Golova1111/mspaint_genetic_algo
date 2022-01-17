import math
import time

import numpy as np

from numba import cuda

import Figures.Ellipse
from Picture import Picture
from Figures.Rectangle import Rectangle
from Figures.Triangle import Triangle
from Figures.Ellipse import Ellipse
from cuda import _calc_delta, _gen_picture


def score_test():
    c1 = 30 * 6 * 2
    c2 = 30 * 8 * 2

    picture = np.random.randint(low=0, high=255, size=(c1, c2, 3), dtype=np.int16)
    some_image = np.random.randint(low=0, high=255, size=(c1, c2, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)

    # first "compilation" time
    answer = _calc_delta(d_picture, some_image)

    print(" =========== ")
    print("== paralel ")

    start = time.time()
    answer = _calc_delta(d_picture, some_image)
    end = time.time()
    print(answer)
    print(end - start)

    print("== numpy ")
    start = time.time()
    answer = np.sum(np.abs(picture - some_image))
    end = time.time()
    print(answer)
    print(end - start)


def generate_test():
    c = 1
    prev_winner = Picture(size=(360 * c, 480 * c))
    prev_winner.parts = [
    #     Triangle(p1=(50, 50), p2=(100, 100), p3=(50, 150), color=np.array([237, 229, 179]), max_size=(360 * c, 480 * c))
    #     Rectangle(p1=[ 23 * c,  28 * c], p2=[324 * c, 480 * c], color=np.array([237, 229, 179]), max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[ 17 * c,  99 * c], p2=[181 * c, 480 * c], color=np.array([174, 218, 234]), max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[ 88 * c, 183 * c], p2=[179 * c, 392 * c], color=np.array([119, 1, 18]),    max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[278 * c,   0 * c], p2=[360 * c, 480 * c], color=np.array([100, 177, 79]),  max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[  0 * c,   0 * c], p2=[ 14 * c, 480 * c], color=np.array([0, 0, 0]),       max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[216 * c,   0 * c], p2=[278 * c, 143 * c], color=np.array([196, 230, 48]),  max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[139 * c, 148 * c], p2=[181 * c, 423 * c], color=np.array([119, 1, 18]),    max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[102 * c,  28 * c], p2=[222 * c, 141 * c], color=np.array([255, 255, 255]), max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[269 * c,   0 * c], p2=[331 * c, 143 * c], color=np.array([100, 177, 79]),  max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[347 * c,   0 * c], p2=[360 * c, 480 * c], color=np.array([0, 0, 0]),       max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[ 33 * c,  25 * c], p2=[101 * c,  98 * c], color=np.array([231, 129, 48]),  max_size=(360 * c, 480 * c)),
    ]

    FIGURES = 1

    for i in range(FIGURES):
        prev_winner.parts.append(
            Ellipse.gen_random(size=(360 * c, 480 * c))
        )
    for i in range(FIGURES):
        prev_winner.parts.append(
            Triangle.gen_random(size=(360 * c, 480 * c))
        )
    for i in range(FIGURES):
        prev_winner.parts.append(
            Rectangle.gen_random(size=(360 * c, 480 * c))
        )

    # first "compilation" time
    _gen_picture(prev_winner)

    start = time.time()
    _gen_picture(prev_winner)
    end = time.time()
    print(end - start)
    prev_winner.visualize()

    start = time.time()
    prev_winner._old_gen_picture()
    end = time.time()
    print(end - start)
    prev_winner.visualize()


if __name__ == '__main__':
    # score_test()
    # print(" ============== ")
    generate_test()
