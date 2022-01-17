import random

import numpy as np

from Color import Color, get_similar_color, c, get_color
from Figures.Figure import Figure


class Ellipse(Figure):
    MUTATION_POSITION_PROB = 0.25
    MUTATION_COLOR_PROB = 0.1
    MUTATION_RECTANGLE_PROBABILITY = 0.03
    MUTATION_POSITION_SCALE = 15

    CUDA_FIGURE_ID = 2

    def __init__(self, center, a, b, color, max_size, color_delta=0):
        self.center = list(center)
        self.a = a
        self.b = b
        self.color = color
        self.color_delta = color_delta
        self._repr_color = get_color(self.color, self.color_delta)

        self.max_h = max_size[0]
        self.max_w = max_size[1]

        self._repr = np.zeros(10)

    def add_part(self, picture):
        xs = np.linspace(0, self.max_h, num=self.max_h)
        ys = np.linspace(0, self.max_w, num=self.max_w)
        xv, yv = np.meshgrid(ys, xs)

        cx, cy = self.center
        mask = (xv - cx)**2 / (self.a * self.a) + (yv - cy)**2 / (self.b * self.b) < 1
        picture[mask, :] = self.color

        return picture

    def mutate(self):
        deltas = np.random.normal(loc=0, scale=self.MUTATION_POSITION_SCALE, size=4)

        if random.random() < self.MUTATION_POSITION_PROB:
            self.center[0] = int(self.center[0] + deltas[0])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.center[1] = int(self.center[1] + deltas[1])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.a = int(self.a + deltas[2])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.b = int(self.b + deltas[3])

        self.center[0] = max(0, self.center[0])
        self.center[1] = max(0, self.center[1])

        self.center[0] = min(self.max_h, self.center[0])
        self.center[1] = min(self.max_h, self.center[1])

        if self.a == 0:
            self.a = 1
        if self.b == 0:
            self.b = 1

        if random.random() < self.MUTATION_COLOR_PROB:
            self._color_mutate()

        if random.random() < self.MUTATION_RECTANGLE_PROBABILITY:
            return self._rectangle_mutate()

        return self

    def _rectangle_mutate(self):
        from Figures.Rectangle import Rectangle

        return Rectangle(
            p1=(self.center[0] - self.a, self.center[1] - self.b),
            p2=(self.center[0] + self.a, self.center[1] + self.b),
            color=self.color,
            color_delta=self.color_delta,
            max_size=(self.max_h, self.max_w)
        )

    @classmethod
    def gen_random(cls, size):
        h = size[0]
        w = size[1]

        h1 = random.randint(0, h)
        w1 = random.randint(0, w)

        return Ellipse(
            center=(h1, w1),
            a=random.randint(5, h // 2),
            b=random.randint(5, w // 2),
            color=Color.ALL[random.randint(0, Color.ALL.shape[0] - 1)],
            max_size=(h, w)
        )

    def _get_repr(self):
        self._repr[0] = self.CUDA_FIGURE_ID
        self._repr[1] = self.center[0]
        self._repr[2] = self.center[1]
        self._repr[3] = self.a
        self._repr[4] = self.b
        self._repr[5:8] = self._repr_color
        return self._repr

    def __repr__(self):
        return (
            f"Ellipse("
            f"center={self.center}, "
            f"a={self.a}, "
            f"b={self.b}, "
            f"color=np.array([{self.color[0]}, {self.color[1]}, {self.color[2]}]), "
            f"color_delta={self.color_delta}, "
            f"max_size=({self.max_h}, {self.max_w})"
            f")"
        )
