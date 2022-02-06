import random

import numpy as np

IS_EXTENDED_COLOR_SPACE = False


class Color:
    # standard 20-colors pallet

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

    # extended pallet
    # https://i.imgur.com/JjFhF.png

    PINK_EXT = np.array([255, 127, 131])
    RED_EXT = np.array([248, 3, 1])
    DARK_EXT = np.array([124,66,63])
    DARK2_EXT = np.array([124,2,0])
    DARK3_EXT = np.array([61,2,0])
    BLACK_EXT = np.array([0,0,0])

    YELLOW_EXT = np.array([255,255,134])
    YELLOW2_EXT = np.array([254,255,0])
    ORANGE_EXT = np.array([253,131,58])
    ORANGE2_EXT = np.array([253,130,0])
    BROWN_EXT = np.array([129,64,0])
    DIRTY_EXT = np.array([127,128,0])

    GREEN1_EXT = np.array([130,254,126])
    GREEN2_EXT = np.array([128,255,0])
    GREEN3_EXT = np.array([8,252,6])
    DARKGREEN1_EXT = np.array([0,128,2])
    DARKGREEN2_EXT = np.array([0,65,0])
    DIRTY2_EXT = np.array([130,127,68])

    GREEN4_EXT = np.array([0,255,133])
    GREEN5_EXT = np.array([7,250,71])
    GREEN6_EXT = np.array([0,126,131])
    GREEN7_EXT = np.array([0,128,66])
    BLACK1_EXT = np.array([0,64,65])
    GRAY1_EXT = np.array([128,128,128])

    BLUE_EXT = np.array([124,255,255])
    BLUE2_EXT = np.array([0,255,255])
    DARKBLUE_EXT = np.array([0,65,133])
    DARKBLUE2_EXT = np.array([1,0,252])
    VIOLET_EXT = np.array([1,0,130])
    GREEN8_EXT = np.array([65,128,127])

    BLUE3_EXT = np.array([0,128,255])
    BLUE4_EXT = np.array([2,127,190])
    BLUE5_EXT = np.array([126,131,251])
    BLUE6_EXT = np.array([0,0,161])
    BLACK3_EXT = np.array([0,0,63])
    GRAY2_EXT = np.array([192, 192, 192])

    PINK1_EXT = np.array([255,127,189])
    VIOLET2_EXT = np.array([128,127,195])
    DARKRED1_EXT = np.array([129,1,59])
    VIOLET3_EXT = np.array([127,0,129])
    BLACK4_EXT = np.array([62,0,68])
    BLACK5_EXT = np.array([61,1,64])

    PINK2_EXT = np.array([251,130,250])
    PINK3_EXT = np.array([251,2,254])
    PINK4_EXT = np.array([253,2,126])
    BLUE7_EXT = np.array([126,2,249])
    DARKBLUE3_EXT = np.array([67,0,123])
    WHITE_EXT = np.array([255,255,255])

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

    _ALL_MAIN_LIST = [
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

    _ALL_EXTENDED_LIST = [
        PINK_EXT,
        RED_EXT,
        DARK_EXT,
        DARK2_EXT,
        DARK3_EXT,
        BLACK_EXT,
        YELLOW_EXT,
        YELLOW2_EXT,
        ORANGE_EXT,
        ORANGE2_EXT,
        BROWN_EXT,
        DIRTY_EXT,
        GREEN1_EXT,
        GREEN2_EXT,
        GREEN3_EXT,
        DARKGREEN1_EXT,
        DARKGREEN2_EXT,
        DIRTY2_EXT,
        GREEN4_EXT,
        GREEN5_EXT,
        GREEN6_EXT,
        GREEN7_EXT,
        BLACK1_EXT,
        GRAY1_EXT,
        BLUE_EXT,
        BLUE2_EXT,
        DARKBLUE_EXT,
        DARKBLUE2_EXT,
        VIOLET_EXT,
        GREEN8_EXT,
        BLUE3_EXT,
        BLUE4_EXT,
        BLUE5_EXT,
        BLUE6_EXT,
        BLACK3_EXT,
        GRAY2_EXT,
        PINK1_EXT,
        VIOLET2_EXT,
        DARKRED1_EXT,
        VIOLET3_EXT,
        BLACK4_EXT,
        BLACK5_EXT,
        PINK2_EXT,
        PINK3_EXT,
        PINK4_EXT,
        BLUE7_EXT,
        DARKBLUE3_EXT,
        WHITE_EXT
    ]

    ALL = np.stack(_ALL_MAIN_LIST + _ALL_EXTENDED_LIST)


c = Color()


def get_color(color, delta):
    amount = delta * 5
    return [
        min(max(0, color[0] + amount), 255),
        min(max(0, color[1] + amount), 255),
        min(max(0, color[2] + amount), 255),
    ]


color_hue_dict = {}
color_hue_reverse_dict = {}

black_discriminate = bd = 32

if not IS_EXTENDED_COLOR_SPACE:
    possible_color_delta_space = range(0, 1)
else:
    possible_color_delta_space = range(-50, 50)

for color in (c._ALL_EXTENDED_LIST + c._ALL_MAIN_LIST):
    for delta in possible_color_delta_space:
        hue = get_color(color, delta)
        if not (hue[0] == hue[1] == hue[2]):
            if hue[0] < bd and hue[1] < bd:
                continue
            if hue[2] < bd and hue[1] < bd:
                continue
            if hue[0] < bd and hue[2] < bd:
                continue

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
