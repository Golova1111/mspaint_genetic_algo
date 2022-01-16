import random

import numpy as np

from Color import Color, get_similar_color, c


class Rectangle:
    MUTATION_POSITION_PROB = 0.25
    MUTATION_COLOR_PROB = 0.1
    MUTATION_POSITION_SCALE = 15

    def __init__(self, p1, p2, color, max_size):
        self.p1 = list(p1)
        self.p2 = list(p2)
        self.color = color

        self.max_h = max_size[0]
        self.max_w = max_size[1]

    def add_part(self, picture):
        picture[
            min(max(0, self.p1[0]), picture.shape[0]) : min(max(0, self.p2[0]), picture.shape[0]),  # h
            min(max(0, self.p1[1]), picture.shape[1]) : min(max(0, self.p2[1]), picture.shape[1]),  # w
            :
        ] = self.color
        return picture

    def mutate(self):
        deltas = np.random.normal(loc=0, scale=self.MUTATION_POSITION_SCALE, size=4)

        if random.random() < self.MUTATION_POSITION_PROB:
            self.p1[0] = int(self.p1[0] + deltas[0])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p1[1] = int(self.p1[1] + deltas[1])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p2[0] = int(self.p2[0] + deltas[2])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p2[1] = int(self.p2[1] + deltas[3])

        self.p1[0] = max(0, self.p1[0])
        self.p2[0] = max(0, self.p2[0])
        self.p1[1] = max(0, self.p1[1])
        self.p2[1] = max(0, self.p2[1])

        self.p1[0] = min(self.max_h, self.p1[0])
        self.p2[0] = min(self.max_h, self.p2[0])
        self.p1[1] = min(self.max_w, self.p1[1])
        self.p2[1] = min(self.max_w, self.p2[1])

        if random.random() < self.MUTATION_COLOR_PROB:
            self.color = get_similar_color(self.color)

        return self

    @classmethod
    def gen_random(cls, size):
        h = size[0]
        w = size[1]

        h1 = random.randint(0, h)
        h2 = random.randint(0, h)
        w1 = random.randint(0, w)
        w2 = random.randint(0, w)

        return Rectangle(
            p1=(min(h1, h2), min(w1, w2)),
            p2=(max(h1, h2), max(w1, w2)),
            color=Color.ALL[random.randint(0, Color.ALL.shape[0] - 1)],
            max_size=(h, w)
        )

    def __repr__(self):
        return (
            f"{self.p1}, {self.p2}, {c.color_dict[tuple(self.color)]}"
        )