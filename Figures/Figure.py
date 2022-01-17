import random

from Color import get_similar_color, get_color


class Figure:
    def _color_mutate(self):
        if random.random() < 0.6:
            self.color_delta += random.randint(-1, 1)
            self.color_delta = min(self.color_delta, 3)
            self.color_delta = max(-3, self.color_delta)
        else:
            self.color = get_similar_color(self.color)
            self.color_delta = random.randint(-3, 3)

        self._repr_color = get_color(self.color, self.color_delta)
