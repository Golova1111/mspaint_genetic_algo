import random

import numpy as np

from Color import get_similar_color, get_color


class Figure:
    MUTATION_ROTATION_SCALE = 0.1

    def _color_mutate(self):
        self.color, self.color_delta = get_similar_color(self.color)
        self._repr_color = get_color(self.color, self.color_delta)

    def _angle_mutate(self):
        self.angle += np.random.normal(loc=0, scale=self.MUTATION_ROTATION_SCALE)
