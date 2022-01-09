from Picture import Color, Picture, Rectangle, get_similar_color

c = Color()


def test_color_similarity():
    for i in range(100):
        similar_color = get_similar_color(Color.RED)
        print(c.color_dict[tuple(similar_color)])


def test_picture():
    p = Picture(size=(240, 320))

    p.parts.append(
        Rectangle(p1=(100, 100), p2=(200, 200), color=c.RED),
    )
    p.parts.append(
        Rectangle(p1=(0, 0), p2=(100, 200), color=c.DARKGREEN),
    )

    p.visualize()

    p.mutate()
    p.visualize()


def main():
    test_picture()


if __name__ == '__main__':
    main()
