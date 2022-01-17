import random

import numpy as np

from Color import Color, get_similar_color, c, get_color
from Figures.Figure import Figure


class Triangle(Figure):
    MUTATION_POSITION_PROB = 0.25
    MUTATION_COLOR_PROB = 0.1
    MUTATION_POSITION_SCALE = 15

    CUDA_FIGURE_ID = 1

    def point_in_triangle(self, point):
        """Returns True if the point is inside the triangle
        and returns False if it falls outside.
        - The argument *point* is a tuple with two elements
        containing the X,Y coordinates respectively.
        - The argument *triangle* is a tuple with three elements each
        element consisting of a tuple of X,Y coordinates.

        It works like this:
        Walk clockwise or counterclockwise around the triangle
        and project the point onto the segment we are crossing
        by using the dot product.
        Finally, check that the vector created is on the same side
        for each of the triangle's segments.
        """
        # Unpack arguments
        x, y = point
        ax, ay = self.p1
        bx, by = self.p2
        cx, cy = self.p3
        # Segment A to B
        side_1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
        # Segment B to C
        side_2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
        # Segment C to A
        side_3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
        # All the signs must be positive or all negative
        return (side_1 < 0.0) == (side_2 < 0.0) == (side_3 < 0.0)

    def __init__(self, p1, p2, p3, color, max_size, color_delta=0):
        self.p1 = list(p1)
        self.p2 = list(p2)
        self.p3 = list(p3)
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

        ax, ay = self.p1
        bx, by = self.p2
        cx, cy = self.p3

        # Segment A to B
        side_1 = (xv - bx) * (ay - by) - (ax - bx) * (yv - by) > 0
        # Segment B to C
        side_2 = (xv - cx) * (by - cy) - (bx - cx) * (yv - cy) > 0
        # Segment C to A
        side_3 = (xv - ax) * (cy - ay) - (cx - ax) * (yv - ay) > 0

        mask = np.all([
            (side_1 == side_2),
            (side_2 == side_3),
            (side_1 == side_3)
        ], axis=(0,))
        picture[mask, :] = self.color

        return picture

    def mutate(self):
        deltas = np.random.normal(loc=0, scale=self.MUTATION_POSITION_SCALE, size=6)

        if random.random() < self.MUTATION_POSITION_PROB:
            self.p1[0] = int(self.p1[0] + deltas[0])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p1[1] = int(self.p1[1] + deltas[1])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p2[0] = int(self.p2[0] + deltas[2])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p2[1] = int(self.p2[1] + deltas[3])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p3[0] = int(self.p3[0] + deltas[4])
        if random.random() < self.MUTATION_POSITION_PROB:
            self.p3[1] = int(self.p3[1] + deltas[5])

        # self.p1[0] = max(0, self.p1[0])
        # self.p2[0] = max(0, self.p2[0])
        # self.p3[0] = max(0, self.p3[0])
        # self.p1[1] = max(0, self.p1[1])
        # self.p2[1] = max(0, self.p2[1])
        # self.p3[1] = max(0, self.p3[1])
        #
        # self.p1[0] = min(self.max_h, self.p1[0])
        # self.p2[0] = min(self.max_h, self.p2[0])
        # self.p3[0] = min(self.max_h, self.p3[0])
        # self.p1[1] = min(self.max_w, self.p1[1])
        # self.p2[1] = min(self.max_w, self.p2[1])
        # self.p3[1] = min(self.max_w, self.p3[1])

        if random.random() < self.MUTATION_COLOR_PROB:
            self._color_mutate()

        return self

    @classmethod
    def gen_random(cls, size):
        h = size[0]
        w = size[1]

        h1 = random.randint(0, h)
        h2 = random.randint(0, h)
        h3 = random.randint(0, h)
        w1 = random.randint(0, w)
        w2 = random.randint(0, w)
        w3 = random.randint(0, w)

        return Triangle(
            p1=(h1, w1),
            p2=(h2, w2),
            p3=(h3, w3),
            color=Color.ALL[random.randint(0, Color.ALL.shape[0] - 1)],
            max_size=(h, w)
        )

    def _get_repr(self):
        self._repr[0] = self.CUDA_FIGURE_ID
        self._repr[1] = self.p1[0]
        self._repr[2] = self.p1[1]
        self._repr[3] = self.p2[0]
        self._repr[4] = self.p2[1]
        self._repr[5] = self.p3[0]
        self._repr[6] = self.p3[1]
        self._repr[7:10] = self._repr_color
        return self._repr

    def __repr__(self):
        return (
            f"Triangle("
            f"p1={self.p1}, "
            f"p2={self.p2}, "
            f"p3={self.p3}, "
            f"color=np.array([{self.color[0]}, {self.color[1]}, {self.color[2]}]), "
            f"color_delta={self.color_delta}, "
            f"max_size=({self.max_h}, {self.max_w})"
            f")"
        )
