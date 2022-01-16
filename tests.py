from Picture import Picture
from Color import Color, get_similar_color
from Rectangle import Rectangle

c = Color()


def test_color_similarity():
    for i in range(100):
        similar_color = get_similar_color(Color.RED)
        print(c.color_dict[tuple(similar_color)])


def test_picture():
    p = Picture(size=(240, 320))

    p.parts.append(
        # Rectangle(p1=(200, 200), p2=(100, 100), color=c.RED),
        Rectangle.gen_random(size=(240, 320))
    )
    # p.parts.append(
    #     Rectangle(p1=(0, 0), p2=(100, 200), color=c.DARKGREEN),
    # )

    p.gen_picture()
    p.visualize()

    # p.mutate()
    # p.visualize()


def main():
    test_picture()


if __name__ == '__main__':
    main()
