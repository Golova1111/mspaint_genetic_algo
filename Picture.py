import copy
import datetime
import pickle
import random

from matplotlib import pyplot as plt

import numpy as np

from Figures.Ellipse import Ellipse
from Figures.Line import Line
from Figures.Rectangle import Rectangle
from Figures.Triangle import Triangle
from cuda import _calc_delta, _gen_picture


class Picture:
    MUTATION_SWAP_PROBABILITY = 0.15
    FIGURE_ADD_PROBABILITY = FIGURE_REMOVE_PROBABILITY = 0.25

    def __init__(self, size, d_picture, max_fignum=100):
        self.size = size
        self.w = size[0]
        self.h = size[1]

        self.max_fignum = max_fignum

        self._score = None
        self.parts = []
        self.d_picture = d_picture

        self.picture = np.zeros(
            (self.w, self.h, 3), dtype=np.int16
        ) + 255

    def gen_picture(self):
        self.picture[:, :, :] = 255
        _gen_picture(self)
        return self.picture

    def _old_gen_picture(self):
        self.picture[:, :, :] = 255

        for elem in self.parts:
            self.picture = elem.add_part(self.picture)

        self._score = None
        return self.picture

    def visualize(self, title=None, is_save=True):
        resize_c = 3
        resized_parts = [fig.resize(resize_c) for fig in self.parts]

        big_pic = Picture(
            size=(self.w * resize_c, self.h * resize_c),
            d_picture=self.d_picture,
            max_fignum=self.max_fignum
        )
        big_pic.parts = resized_parts
        big_pic.gen_picture()

        plt.imshow(big_pic.picture, interpolation='nearest')

        # plt.imshow(self.picture, interpolation='nearest')
        if title:
            plt.title(title)
        if is_save:
            plt.savefig(f"/home/vadym/University/Term 3/EvolAlg/Project/pic/save/epoch{len(self.parts)}_{datetime.datetime.now()}.png")
        plt.show()

    def delta(self, icon_picture):
        # self._score = np.sum(np.abs(icon_picture - self.picture))
        return np.sum(np.abs(icon_picture - self.picture))

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
                if random.random() < 0.5:
                    position = len(self.parts)
                else:
                    position = random.randint(0, len(self.parts))

            if len(self.parts) > 25:
                is_small = 3
            elif len(self.parts) > 15:
                is_small = 2
            elif len(self.parts) > 8:
                is_small = 1.5
            else:
                is_small = 0

            # 45% -- Rectangle
            # 25% -- Triangle
            # 25% -- Ellipse
            # 5%  -- Line (Rectangle)

            prob = random.random()
            if prob < 0.45:
                self.parts.insert(
                    position,
                    Rectangle.gen_random(size=self.size, is_small=is_small)
                )
            elif prob < 0.5:
                self.parts.insert(
                    position,
                    Line.gen_random(size=self.size, is_small=is_small)
                )
            elif prob < 0.75:
                self.parts.insert(
                    position,
                    Triangle.gen_random(size=self.size, is_small=is_small)
                )
            else:
                self.parts.insert(
                    position,
                    Ellipse.gen_random(size=self.size, is_small=is_small)
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
        figsize = len(self.parts)
        if figsize < 15:
            k = random.randint(1, min(4, figsize))
        elif figsize < 30:
            k = random.randint(1, 3)
        else:
            k = random.randint(1, 2)

        if len(self.parts) < 10:
            mutate_elem_idx = random.choices(range(figsize), k=k)
        else:
            if random.random() < 0.75:
                mutate_elem_idx = np.random.exponential(scale=figsize / 2, size=k)
                mutate_elem_idx = np.round(np.abs(mutate_elem_idx - figsize)).astype(int)
                mutate_elem_idx = np.minimum(mutate_elem_idx, np.zeros(shape=k) + (figsize - 1)).astype(int)
            else:
                mutate_elem_idx = random.choices(range(figsize), k=k)

        for idx in mutate_elem_idx:
            self.parts[idx] = self.parts[idx].mutate()

        if random.random() < self.MUTATION_SWAP_PROBABILITY:
            self._swap_random(gp=False)

        if random.random() < self.FIGURE_REMOVE_PROBABILITY:
            self.remove_random_figure(gp=False)

        if random.random() < self.FIGURE_ADD_PROBABILITY:
            self.add_random_figure(gp=False)

        if gp:
            self.gen_picture()
        return self

    @classmethod
    def full_mutate(cls, pic, gp=True):
        new_pic = Picture(size=pic.size, d_picture=pic.d_picture)
        new_pic.parts = pickle.loads(pickle.dumps(pic.parts))
        new_pic.max_fignum = pic.max_fignum
        return new_pic.mutate(gp=gp)

    @classmethod
    def generate_default(cls, picture, d_picture, max_fignum):
        START_FIG_NUMBER = max_fignum

        p = cls(size=picture.shape[:2], max_fignum=max_fignum, d_picture=d_picture)
        for i in range(START_FIG_NUMBER):
            p.add_random_figure(gp=False)

        p.gen_picture()
        return p

    @classmethod
    def generate_similar(cls, icon_picture, max_fignum):
        p = cls(size=icon_picture.size, max_fignum=max_fignum, d_picture=icon_picture.d_picture)
        p.parts = pickle.loads(pickle.dumps(icon_picture.parts))

        if max_fignum - len(icon_picture.parts) > 0:
            random_add_count = random.randint(
                0, max_fignum - len(icon_picture.parts)
            )

            for i in range(random_add_count):
                p.add_random_figure(gp=False)

        p.mutate(gp=False)
        p.gen_picture()
        return p
