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


def test_picture_color_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    col = c.GRAY1

    p.parts.append(
        Rectangle(p1=(100, 100), p2=(200, 200), angle=0, color=col, max_size=(240, 320)),
    )

    for x in range(10):
        for y in range(10):
            fig = Rectangle(p1=(10 * x, 10 * y), p2=(10 * (x + 1), 10 * (y + 1)), angle=0, color=col, max_size=(240, 320))
            fig._color_mutate()
            p.parts.append(fig)

    p.gen_picture()
    p.visualize(is_save=False)


def test_picture_rectangle_generation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    for x in range(10):
        f = Rectangle.gen_random(size=(240, 320))
        f.color = c.RED
        f._color_mutate()

        p.parts.append(f)
        print(f)

    p.gen_picture()
    p.visualize(is_save=False)


def test_picture_ellipse_generation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    for x in range(50):
        f = Ellipse.gen_random(size=(240, 320))
        f.color = c.RED
        f._color_mutate()

        p.parts.append(f)
        print(f)

    p.gen_picture()
    p.visualize(is_save=False)


def test_picture_triangle_generation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    for x in range(50):
        f = Triangle.gen_random(size=(240, 320), is_small=3)
        f.color = c.RED
        f._color_mutate()

        p.parts.append(f)
        print(f)

    p.gen_picture()
    p.visualize(is_save=False)


def test_triangle_rectangle_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    p.parts.append(
        Triangle(p1=(100, 100), p2=(200, 200), p3=(50, 200), color=c.LIGHTBLUE, max_size=(240, 320)),
    )

    p.gen_picture()
    p.visualize(is_save=False)

    p.parts[0] = p.parts[0]._rectangle_mutate()
    print(p.parts[0])

    p.gen_picture()
    p.visualize(is_save=False)


def test_ellipse_triangle_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    p.parts.append(
        Ellipse(center=(100, 100), a=60, b=30, angle=1.5, color=c.LIGHTBLUE, max_size=(240, 320)),
    )

    p.gen_picture()
    p.visualize(is_save=False)

    p.parts[0] = p.parts[0]._triangle_mutate()
    print(p.parts[0])

    p.gen_picture()
    p.visualize(is_save=False)


def test_rectangle_triangle_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    p.parts.append(
        Rectangle(p1=(50, 50), p2=(120, 200), angle=1, color=c.LIGHTBLUE, max_size=(240, 320)),
    )

    p.gen_picture()
    p.visualize(is_save=False)

    p.parts[0] = p.parts[0]._triangle_mutate()
    print(p.parts[0])

    p.gen_picture()
    p.visualize(is_save=False)


def color_test():
    picture = np.zeros(shape=(321, 321, 3), dtype=np.float)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(321, 321), d_picture=d_picture)

    i = 0

    try:
        for y in range(0, 32):
            for x in range(0, 32):
                # color, color_delta = all_colors_component_list[-i]
                # print(color)

                f = Rectangle.gen_random(size=(320, 320))
                f.p1 = [10 * x, 10 * y]
                f.p2 = [10 * (x + 1), 10 * (y + 1)]
                f.angle = 0

                p.parts.append(f)
                # print(f)

                i += 1
    except Exception as ex:
        print(ex)

    p.gen_picture()
    p.visualize(is_save=False)


def main():
    # test_picture_triangle_generation()
    # test_triangle_rectangle_mutation()
    # test_ellipse_triangle_mutation()
    # test_rectangle_triangle_mutation()
    # color_test()
    test_picture_color_mutation()


if __name__ == '__main__':
    main()
