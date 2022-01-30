import random

import numpy as np

from Color import get_similar_color, get_color


class Figure:
    MUTATION_ROTATION_SCALE = 0.2

    def _color_mutate(self):
        self.color, self.color_delta = get_similar_color(self.color)
        self._repr_color = get_color(self.color, self.color_delta)

    def _angle_mutate(self):
        delta = np.random.uniform(-self.MUTATION_ROTATION_SCALE, self.MUTATION_ROTATION_SCALE)
        # print(f"{self.angle} ==> {self.angle + delta}")
        self.angle = self.angle + delta
