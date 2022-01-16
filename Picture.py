import copy
import datetime
import random

from matplotlib import pyplot as plt

import numpy as np

from Figures.Ellipse import Ellipse
from Figures.Rectangle import Rectangle
from Figures.Triangle import Triangle
from cuda import _calc_delta


class Picture:
    MUTATION_SWAP_PROBABILITY = 0.15
    FIGURE_ADD_PROBABILITY = FIGURE_REMOVE_PROBABILITY = 0.25

    def __init__(self, size, picture=None, max_fignum=100):
        self.size = size
        self.w = size[0]
        self.h = size[1]

        self.max_fignum = max_fignum

        self._score = None
        self.parts = []

        self.picture = np.zeros(
            (self.w, self.h, 3), dtype=np.int16
        ) + 255

    def gen_picture(self):
        self.picture[:, :, :] = 255

        for elem in self.parts:
            self.picture = elem.add_part(self.picture)

        self._score = None
        return self.picture

    def visualize(self, title=None):
        plt.imshow(self.picture, interpolation='nearest')
        if title:
            plt.title(title)
        plt.savefig(f"/home/vadym/University/Term 3/EvolAlg/Project/pic/save/epoch{len(self.parts)}_{datetime.datetime.now()}.png")
        plt.show()

    def delta(self, icon_picture):
        self._score = np.sum(np.abs(icon_picture - self.picture))
        return self._score

    def score(self, d_icon_picture):
        if self._score:
            return self._score
        else:
            self._score = _calc_delta(image=self.picture, device_pic=d_icon_picture)
            return self._score

    def _swap_random(self, gp=True):
        if len(self.parts) > 1:
            idx = range(len(self.parts))
            i1, i2 = random.sample(idx, 2)
            self.parts[i1], self.parts[i2] = self.parts[i2], self.parts[i1]

        if gp:
            self.gen_picture()

    def add_random_figure(self, gp=True, last=False):
        if len(self.parts) < self.max_fignum:
            if last:
                position = len(self.parts)
            else:
                position = random.randint(0, len(self.parts))

            if random.random() < 0.5:
                self.parts.insert(
                    position,
                    Rectangle.gen_random(size=self.size)
                )
            elif random.random() < 0.5:
                self.parts.insert(
                    position,
                    Triangle.gen_random(size=self.size)
                )
            else:
                self.parts.insert(
                    position,
                    Ellipse.gen_random(size=self.size)
                )

            if gp:
                self.gen_picture()

    def remove_random_figure(self, gp=True):
        if len(self.parts) > 1:
            position = random.randint(0, len(self.parts) - 1)
            self.parts.pop(position)

        if gp:
            self.gen_picture()

    def mutate(self, gp=True):
        for elem in random.choices(self.parts, k=random.randint(1, min(3, len(self.parts)))):
            elem.mutate()

        if random.random() < self.MUTATION_SWAP_PROBABILITY:
            self._swap_random(gp=False)

        if random.random() < self.FIGURE_ADD_PROBABILITY:
            self.add_random_figure(gp=False)

        if random.random() < self.FIGURE_REMOVE_PROBABILITY:
            self.remove_random_figure(gp=False)

        if gp:
            self.gen_picture()
        return self

    @classmethod
    def full_mutate(cls, pic):
        new_pic = Picture(size=pic.size, picture=pic)
        new_pic.parts = copy.deepcopy(pic.parts)
        new_pic.max_fignum = pic.max_fignum
        return new_pic.mutate()

    @classmethod
    def generate_default(cls, picture, max_fignum):
        START_FIG_NUMBER = max_fignum

        p = cls(size=picture.shape[:2], max_fignum=max_fignum)
        for i in range(START_FIG_NUMBER):
            p.add_random_figure(gp=False)

        p.gen_picture()
        return p

    @classmethod
    def generate_similar(cls, icon_picture, max_fignum):
        p = cls(size=icon_picture.size, max_fignum=max_fignum)
        p.parts = copy.deepcopy(icon_picture.parts)

        for i in range(max_fignum - len(icon_picture.parts)):
            p.add_random_figure(gp=False, last=True)

        for i in range(random.randint(0, max_fignum)):
           p.mutate(gp=False)

        p.gen_picture()
        return p
