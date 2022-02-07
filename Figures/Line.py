import math
import random

from Color import get_random_color
from Figures.Rectangle import Rectangle


class Line(Rectangle):
    def __init__(self, p1, l, color, max_size, angle, color_delta=0, width=2):
        p2 = p1[0] + width, p1[1] + l
        super(Line, self).__init__(p1, p2, max_size, color, angle, color_delta)

    @classmethod
    def gen_random(cls, size, is_small=False):
        h = size[1]
        w = size[0]

        width = random.randint(1, 3)
        h1 = random.randint(0, h)
        w1 = random.randint(0, w)

        color, color_delta = get_random_color()
        l = random.randint(0, max(h, w) // 2)

        if is_small:
            l = l // is_small

        angle = (random.random() - 0.5) * (2 * math.pi)

        return Rectangle(
            p1=(h1, w1),
            p2=(h1 + width, w1 + l),
            angle=angle,
            color=color,
            color_delta=color_delta,
            max_size=(h, w)
        )
