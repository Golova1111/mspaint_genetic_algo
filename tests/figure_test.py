import numpy as np
from numba import cuda

from Figures.Ellipse import Ellipse
from Picture import Picture
from Color import Color, get_similar_color
from Figures.Rectangle import Rectangle
from Figures.Triangle import Triangle

c = Color()


def test_color_similarity():
    for i in range(100):
        similar_color = get_similar_color(Color.RED)
        print(c.color_dict[tuple(similar_color)])


def test_picture():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    # p.parts.append(
    #     # Rectangle(p1=(200, 200), p2=(100, 100), color=c.RED),
    #     Rectangle.gen_random(size=(240, 320))
    # )
    for i in range(10):
        p.parts = [
            Rectangle(p1=(0, 100), p2=(200, 200), angle=0.2 * i, color=c.RED, max_size=(240, 320)),
            # Triangle(p1=(50, 50), p2=(100, 100), p3=(50, 150), color=c.RED, max_size=(240, 320))
            # Ellipse(center=(100, 100), a=10, b=50, color=c.RED, max_size=(240, 320), angle=0)
        ]
    # p.parts.append(
    #     Rectangle(p1=(0, 0), p2=(100, 200), color=c.DARKGREEN),
    # )

        p.gen_picture()
        p.visualize(is_save=False)

    # print(p.parts[0].color)
    # print(p.parts[0].color_delta)
    # p.parts[0]._color_mutate()
    # print( ' ---------- ')
    # print(p.parts[0].color)
    # print(p.parts[0].color_delta)
    # p.parts[0].color_delta = -3
    # p.gen_picture()
    # p.mutate()

    p.parts[0]._angle_mutate()
    # p.parts[0].mutate()
    p.gen_picture()
    p.visualize(is_save=False)

    # p.parts[0].color_delta = 3
    # p.gen_picture()
    # p.mutate()
    # p.visualize(is_save=False)


def main():
    test_picture()


if __name__ == '__main__':
    main()
