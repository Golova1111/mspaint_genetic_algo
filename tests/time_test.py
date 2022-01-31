import math
import random
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
    picture = np.zeros(shape=(360 * c, 480 * c, 3), dtype=np.int16) + 50
    d_picture = cuda.to_device(picture)
    prev_winner = Picture(size=(360 * c, 480 * c), d_picture=d_picture)
    prev_winner.parts = [
         Triangle(p1=(50, 50), p2=(100, 100), p3=(50, 150), color=np.array([237, 229, 179]), max_size=(360 * c, 480 * c)),
         Rectangle(p1=[ 28 * c,  23 * c,], p2=[480 * c, 324 * c], color=np.array([237, 229, 179]), max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[ 99 * c,  17 * c,], p2=[480 * c, 181 * c], color=np.array([174, 218, 234]), max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[183 * c,  88 * c,], p2=[392 * c, 179 * c], color=np.array([119, 1, 18]),    max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[  0 * c, 278 * c,], p2=[480 * c, 360 * c], color=np.array([100, 177, 79]),  max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[  0 * c,   0 * c,], p2=[480 * c,  14 * c], color=np.array([0, 0, 0]),       max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[  0 * c, 216 * c,], p2=[143 * c, 278 * c], color=np.array([196, 230, 48]),  max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[148 * c, 139 * c,], p2=[423 * c, 181 * c], color=np.array([119, 1, 18]),    max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[ 28 * c, 102 * c,], p2=[141 * c, 222 * c], color=np.array([255, 255, 255]), max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[  0 * c, 269 * c,], p2=[143 * c, 331 * c], color=np.array([100, 177, 79]),  max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[  0 * c, 347 * c,], p2=[480 * c, 360 * c], color=np.array([0, 0, 0]),       max_size=(360 * c, 480 * c), angle=0),
         Rectangle(p1=[ 25 * c,  33 * c,], p2=[ 98 * c, 101 * c], color=np.array([231, 129, 48]),  max_size=(360 * c, 480 * c), angle=0),
    ]

    # FIGURES = 10
    #
    # for i in range(FIGURES):
    #     prev_winner.parts.append(
    #         Ellipse.gen_random(size=(360 * c, 480 * c))
    #     )
    # for i in range(FIGURES):
    #     prev_winner.parts.append(
    #         Triangle.gen_random(size=(360 * c, 480 * c))
    #     )
    # for i in range(FIGURES):
    #     prev_winner.parts.append(
    #         Rectangle.gen_random(size=(360 * c, 480 * c))
    #     )

    # first "compilation" time
    _gen_picture(prev_winner)

    start = time.time()
    _gen_picture(prev_winner)
    end = time.time()
    print("cuda")
    print(end - start)
    prev_winner.visualize(is_save=False)

    # print("score:", prev_winner.score(d_icon_picture=d_picture))
    # print("delta:", prev_winner.delta(icon_picture=picture))

    # start = time.time()
    # prev_winner._old_gen_picture()
    # end = time.time()
    # print("old style")
    # print(end - start)
    # prev_winner.visualize()



if __name__ == '__main__':
    # score_test()
    # print(" ============== ")
    generate_test()
