import copy
import random

from numpy.random import choice
from matplotlib import pyplot as plt


import numpy as np


class Picture:
    MUTATION_SWAP_PROBABILITY = 0.15
    FIGURE_ADD_PROBABILITY = FIGURE_REMOVE_PROBABILITY = 0.2

    def __init__(self, size, picture=None):
        self.size = size
        self.w = size[0]
        self.h = size[1]

        self.parts = []
        if not picture:
            self.background_color = Color.ALL[
                np.random.choice(Color.ALL.shape[0])
            ]
        else:
            self.background_color = picture.background_color

        self.picture = np.zeros(
            (self.w, self.h, 3), dtype=np.int16
        ) + 255  # self.background_color

    def gen_picture(self):
        self.picture = np.ones(
            (self.w, self.h, 3), dtype=np.int16
        ) + 255  # + self.background_color

        for elem in self.parts:
            self.picture = elem.add_part(self.picture)

        return self.picture

    def visualize(self, title=None):
        plt.imshow(self.picture, interpolation='nearest')
        if title:
            plt.title(title)
        plt.show()

    def delta(self, icon_picture):
        return np.sum(np.abs(icon_picture - self.picture))

    def _swap_random(self):
        if len(self.parts) > 1:
            idx = range(len(self.parts))
            i1, i2 = random.sample(idx, 2)
            self.parts[i1], self.parts[i2] = self.parts[i2], self.parts[i1]

    def add_random_figure(self):
        position = random.randint(0, len(self.parts))
        self.parts.insert(
            position,
            Rectangle(
                p1=(random.randint(0, self.h), random.randint(0, self.w)),
                p2=(random.randint(0, self.h), random.randint(0, self.w)),
                color=Color.ALL[random.randint(0, Color.ALL.shape[0] - 1)]
            )
        )

        self.gen_picture()

    def remove_random_figure(self):
        if len(self.parts) > 1:
            position = random.randint(0, len(self.parts) - 1)
            self.parts.pop(position)

    def mutate(self):
        for elem in self.parts:
            elem.mutate()

        if random.random() < self.MUTATION_SWAP_PROBABILITY:
            self._swap_random()

        if random.random() < self.FIGURE_ADD_PROBABILITY:
            self.add_random_figure()

        if random.random() < self.FIGURE_REMOVE_PROBABILITY:
            self.remove_random_figure()

        if random.random() < 0:
            self.background_color = get_similar_color(
                self.background_color
            )

        self.gen_picture()
        return self

    @classmethod
    def full_mutate(cls, pic):
        new_pic = Picture(size=pic.size, picture=pic)
        new_pic.parts = copy.deepcopy(pic.parts)
        return new_pic.mutate()

    @classmethod
    def generate_default(cls, picture):
        START_FIG_NUMBER = 5

        p = cls(size=picture.shape[:2])
        for i in range(START_FIG_NUMBER):
            p.add_random_figure()

        p.gen_picture()
        return p


class Rectangle:
    MUTATION_POSITION_PROB = 0.25
    MUTATION_COLOR_PROB = 0.15
    MUTATION_POSITION_SCALE = 0.15

    def __init__(self, p1, p2, color):
        self.p1 = list(p1)
        self.p2 = list(p2)
        self.color = color

    def add_part(self, picture):
        picture[
            min(max(0, self.p1[0]), picture.shape[0]) : min(max(0, self.p2[0]), picture.shape[0]),  # h
            min(max(0, self.p1[1]), picture.shape[1]) : min(max(0, self.p2[1]), picture.shape[1]),  # w
            :
        ] = self.color
        return picture

    def mutate(self):
        self.p1[0] = int(np.random.normal(loc=1, scale=self.MUTATION_POSITION_SCALE) * self.p1[0])
        self.p1[1] = int(np.random.normal(loc=1, scale=self.MUTATION_POSITION_SCALE) * self.p1[1])
        self.p2[0] = int(np.random.normal(loc=1, scale=self.MUTATION_POSITION_SCALE) * self.p2[0])
        self.p2[1] = int(np.random.normal(loc=1, scale=self.MUTATION_POSITION_SCALE) * self.p2[1])

        self.color = get_similar_color(self.color)

        return self


class Color:
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    GRAY1 = np.array([196, 196, 196])
    GRAY2 = np.array([128, 128, 128])
    DARKRED = np.array([119, 1, 18])
    BROWN = np.array([172, 123, 90])
    RED = np.array([208, 31, 40])
    PINK = np.array([238, 176, 202])
    DARKORANGE = np.array([231, 129, 48])
    ORANGE = np.array([242, 202, 41])
    YELLOW = np.array([251, 241, 42])
    LIGHTYELLOW = np.array([237, 229, 179])
    DARKGREEN = np.array([100, 177, 79])
    LIGHTGREEN = np.array([196, 230, 48])
    BLUE = np.array([87, 163, 231])
    LIGHTBLUE = np.array([174, 218, 234])
    DARKBLUE = np.array([65, 74, 203])
    DIRTYBLUE = np.array([124, 147, 190])
    VIOLET = np.array([147, 75, 164])
    LIGHTVIOLET = np.array([199, 192, 231])

    color_dict = {
        tuple(BLACK): "BLACK",
        tuple(WHITE): "WHITE",
        tuple(GRAY1): "GRAY1",
        tuple(GRAY2): "GRAY2",
        tuple(DARKRED): "DARKRED",
        tuple(BROWN): "BROWN",
        tuple(RED): "RED",
        tuple(PINK): "PINK",
        tuple(DARKORANGE): "DARKORANGE",
        tuple(ORANGE): "ORANGE",
        tuple(YELLOW): "YELLOW",
        tuple(LIGHTYELLOW): "LIGHTYELLOW",
        tuple(DARKGREEN): "DARKGREEN",
        tuple(LIGHTGREEN): "LIGHTGREEN",
        tuple(BLUE): "BLUE",
        tuple(LIGHTBLUE): "LIGHTBLUE",
        tuple(DARKBLUE): "DARKBLUE",
        tuple(DIRTYBLUE): "DIRTYBLUE",
        tuple(VIOLET): "VIOLET",
        tuple(LIGHTVIOLET): "LIGHTVIOLET",
    }

    ALL = np.stack(
        [
            BLACK,
            WHITE,
            GRAY1,
            GRAY2,
            DARKRED,
            BROWN,
            RED,
            PINK,
            DARKORANGE,
            ORANGE,
            YELLOW,
            LIGHTYELLOW,
            DARKGREEN,
            LIGHTGREEN,
            BLUE,
            LIGHTBLUE,
            DARKBLUE,
            DIRTYBLUE,
            VIOLET,
            LIGHTVIOLET
        ]
    )

def get_similar_color(color):
    delta = np.sum(
        np.abs(Color.ALL - color) ** 0.2, axis=1
    )
    delta = np.abs(delta - np.max(delta))
    delta = delta / np.sum(delta)

    return Color.ALL[np.random.choice(Color.ALL.shape[0], p=delta)]