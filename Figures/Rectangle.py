import math
import random

import numpy as np

from Color import Color, get_similar_color, c, get_color, get_random_color
from Figures.Figure import Figure


class Rectangle(Figure):
    MUTATION_POSITION_PROB = 0.15
    MUTATION_COLOR_PROB = 0.15
    MUTATION_POSITION_SCALE = 15
    MUTATION_ROTATE_PROB = 0.15
    MUTATION_ELLIPSE_PROBABILITY = 0.03

    CUDA_FIGURE_ID = 0

    def __init__(self, p1, p2, color, max_size, angle, color_delta=0):
        self.p1 = list(p1)
        self.p2 = list(p2)
        self.color = color
        self.color_delta = color_delta
        self.angle = angle
        self._repr_color = get_color(self.color, self.color_delta)

        self.max_h = max_size[0]
        self.max_w = max_size[1]

        self._repr = np.zeros(10)

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
            self._color_mutate()

        if random.random() < self.MUTATION_ROTATE_PROB:
            self._angle_mutate()

        if random.random() < self.MUTATION_ELLIPSE_PROBABILITY:
            return self._ellipse_mutate()

        return self

    def _ellipse_mutate(self):
        from Figures.Ellipse import Ellipse

        center_x = (self.p1[0] + self.p2[0]) // 2
        center_y = (self.p1[1] + self.p2[1]) // 2

        a = abs(self.p1[0] - self.p2[0]) // 2
        b = abs(self.p1[1] - self.p2[1]) // 2

        return Ellipse(
            center=(center_x, center_y),
            a=a,
            b=b,
            color=self.color,
            color_delta=self.color_delta,
            angle=self.angle,
            max_size=(self.max_h, self.max_w)
        )

    @classmethod
    def gen_random(cls, size, is_small=False):
        h = size[1]
        w = size[0]

        h1 = random.randint(0, h)
        w1 = random.randint(0, w)

        hsize = random.randint(5, h // 2)
        wsize = random.randint(5, w // 2)
        color, color_delta = get_random_color()

        if is_small:
            hsize = hsize // 2
            wsize = wsize // 2

        angle = (random.random() - 0.5) * (2 * math.pi)

        return Rectangle(
            p1=(h1, w1),
            p2=(min(h1 + hsize, w), min(w1 + wsize, w)),
            angle=angle,
            color=color,
            color_delta=color_delta,
            max_size=(h, w)
        )

    def _get_repr(self):
        self._repr[0] = self.CUDA_FIGURE_ID
        self._repr[1] = int(self.p1[0])
        self._repr[2] = int(self.p1[1])
        self._repr[3] = int(self.p2[0])
        self._repr[4] = int(self.p2[1])
        self._repr[5:8] = self._repr_color
        self._repr[9] = self.angle
        return self._repr


    def __repr__(self):
        return (
            f"Rectangle("
            f"p1={self.p1}, "
            f"p2={self.p2}, "
            f"color={self.color}, "
            f"color_delta={self.color_delta}, "
            f"angle={self.angle}, "
            f"max_size=({self.max_h}, {self.max_w})"
            f")"
        )
