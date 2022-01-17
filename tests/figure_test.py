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
    p = Picture(size=(240, 320))

    # p.parts.append(
    #     # Rectangle(p1=(200, 200), p2=(100, 100), color=c.RED),
    #     Rectangle.gen_random(size=(240, 320))
    # )
    p.parts.append(
        # Rectangle(p1=(200, 200), p2=(100, 100), color=c.RED),
        # Triangle(p1=(50, 50), p2=(100, 100), p3=(50, 150), color=c.RED, max_size=(240, 320))
        Ellipse(center=(100, 100), a=50, b=50, color=c.RED, max_size=(240, 320))
    )
    # p.parts.append(
    #     Rectangle(p1=(0, 0), p2=(100, 200), color=c.DARKGREEN),
    # )

    p.gen_picture()
    p.visualize()

    # print(p.parts[0].color)
    # print(p.parts[0].color_delta)
    # p.parts[0]._color_mutate()
    # print( ' ---------- ')
    # print(p.parts[0].color)
    # print(p.parts[0].color_delta)
    p.parts[0].color_delta = -3
    p.gen_picture()
    # p.mutate()
    p.visualize()

    p.parts[0].color_delta = 3
    p.gen_picture()
    # p.mutate()
    p.visualize()


def main():
    test_picture()


if __name__ == '__main__':
    main()
