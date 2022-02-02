import math
import random

import numpy as np

from Color import Color, get_similar_color, c, get_color, get_random_color
from Figures.Figure import Figure


class Ellipse(Figure):
    MUTATION_POSITION_PROB = 0.15
    MUTATION_ROTATE_PROB = 0.15
    MUTATION_COLOR_PROB = 0.15
    MUTATION_RECTANGLE_PROBABILITY = 0.03
    MUTATION_TRIANGLE_PROBABILITY = 0.03

    MUTATION_POSITION_SCALE = 15
    CUDA_FIGURE_ID = 2

    def __init__(self, center, a, b, color, max_size, angle, color_delta=0):
        self.center = list(center)
        self.a = a
        self.b = b
        self.color = color
        self.color_delta = color_delta
        self._repr_color = get_color(self.color, self.color_delta)

        self.max_h = max_size[0]
        self.max_w = max_size[1]
        self.angle = angle

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
        if random.random() < self.MUTATION_TRIANGLE_PROBABILITY:
            return self._triangle_mutate()
        if random.random() < self.MUTATION_RECTANGLE_PROBABILITY:
            return self._rectangle_mutate()

        deltas = np.random.normal(loc=0, scale=self.MUTATION_POSITION_SCALE, size=4)

        if random.random() < self.MUTATION_POSITION_PROB:
            self.center[0] = int(self.center[0] + deltas[0])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.center[1] = int(self.center[1] + deltas[1])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.a = int(self.a + deltas[2])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.b = int(self.b + deltas[3])

        if random.random() < self.MUTATION_ROTATE_PROB:
            self._angle_mutate()

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

        return self

    def _rectangle_mutate(self):
        from Figures.Rectangle import Rectangle

        return Rectangle(
            p1=(self.center[0] - self.a, self.center[1] - self.b),
            p2=(self.center[0] + self.a, self.center[1] + self.b),
            color=self.color,
            color_delta=self.color_delta,
            angle=self.angle,
            max_size=(self.max_h, self.max_w)
        )

    def _triangle_mutate(self):
        from Figures.Triangle import Triangle

        angle1 = random.random() * 2
        angle2 = random.random() * 2 + 2
        angle3 = random.random() * 2 + 4

        # p1 = [self.center[0] + int(self.a * math.cos(angle1)), self.center[1] + int(self.b * math.sin(angle1))]
        # p2 = [self.center[0] + int(self.a * math.cos(angle2)), self.center[1] + int(self.b * math.sin(angle2))]
        # p3 = [self.center[0] + int(self.a * math.cos(angle3)), self.center[1] + int(self.b * math.sin(angle3))]

        p1 = [int(self.a * math.cos(angle1)), int(self.b * math.sin(angle1))]
        p2 = [int(self.a * math.cos(angle2)), int(self.b * math.sin(angle2))]
        p3 = [int(self.a * math.cos(angle3)), int(self.b * math.sin(angle3))]

        asin, acos = math.sin(self.angle), math.cos(self.angle)

        p1 = [int(p1[0] * acos - p1[1] * asin) + self.center[0], int(p1[0] * asin + p1[1] * acos) + self.center[1]]
        p2 = [int(p2[0] * acos - p2[1] * asin) + self.center[0], int(p2[0] * asin + p2[1] * acos) + self.center[1]]
        p3 = [int(p3[0] * acos - p3[1] * asin) + self.center[0], int(p3[0] * asin + p3[1] * acos) + self.center[1]]

        return Triangle(
            p1=p1,
            p2=p2,
            p3=p3,
            color=self.color,
            color_delta=self.color_delta,
            max_size=(self.max_h, self.max_w)
        )

    @classmethod
    def gen_random(cls, size, is_small=False):
        h = size[1]
        w = size[0]

        h1 = random.randint(0, h)
        w1 = random.randint(0, w)
        angle = (random.random() - 0.5) * (2 * math.pi)

        a = random.randint(5, h // 3)
        b = random.randint(5, w // 3)

        if is_small:
            a = a // is_small
            b = b // is_small

        color, color_delta = get_random_color()

        return Ellipse(
            center=(h1, w1),
            a=a,
            b=b,
            angle=angle,
            color=color,
            color_delta=color_delta,
            max_size=(h, w)
        )

    def _get_repr(self):
        self._repr[0] = self.CUDA_FIGURE_ID
        self._repr[1] = self.center[0]
        self._repr[2] = self.center[1]
        self._repr[3] = self.a
        self._repr[4] = self.b
        self._repr[5:8] = self._repr_color
        self._repr[9] = self.angle
        return self._repr

    def __repr__(self):
        return (
            f"Ellipse("
            f"center={self.center}, "
            f"a={self.a}, "
            f"b={self.b}, "
            f"color={self.color}, "
            f"color_delta={self.color_delta}, "
            f"angle={self.angle}, "
            f"max_size=({self.max_h}, {self.max_w})"
            f")"
        )
