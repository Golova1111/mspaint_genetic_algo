import random

import numpy as np


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

    _ALL_LIST = [
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


c = Color()


def get_color(color, delta):
    amount = delta * 20
    return np.array([
        min(max(0, color[0] + amount), 255),
        min(max(0, color[1] + amount), 255),
        min(max(0, color[2] + amount), 255),
    ])


color_hue_dict = {}
color_hue_reverse_dict = {}

for color in c._ALL_LIST:
    for delta in range(-10, 10):
        hue = get_color(color, delta)
        if not color_hue_reverse_dict.get(tuple(hue)):
            color_hue_dict[tuple(color), delta] = hue
            color_hue_reverse_dict[tuple(hue)] = list(color), delta

all_colors = np.stack(list(color_hue_dict.values()))
all_colors_component_list = list(color_hue_reverse_dict.values())


def get_similar_color(color):
    delta = np.sum(
        np.abs(all_colors - color), axis=1
    )
    delta[delta > np.percentile(delta, 50)] = 0
    delta = 1 / delta
    delta[delta == np.inf] = 0

    # delta[delta < np.percentile(delta, 10)] = 0
    delta = delta / np.sum(delta)

    return color_hue_reverse_dict[
        tuple(
            all_colors[np.random.choice(all_colors.shape[0], p=delta)]
        )
    ]


def get_random_color():
    return random.choice(all_colors_component_list)
