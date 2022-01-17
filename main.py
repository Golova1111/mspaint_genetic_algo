import numpy as np
from matplotlib import image

from ga_run import init


# [
#     Ellipse(center=[8, 151], a=27, b=45, color=np.array([100 177  79]), max_size=(180, 240)),
#     Triangle(p1=[81, 38], p2=[8, 64], p3=[6, 20], color=np.array([199 192 231]), max_size=(180, 240)),
#     Rectangle(p1=[10, 42], p2=[37, 240], color=np.array([174 218 234]), max_size=(180, 240)),
#     Ellipse(center=[63, 12], a=13, b=25, color=np.array([255 255 255]), max_size=(180, 240)),
#     Rectangle(p1=[61, 87], p2=[94, 140], color=np.array([119   1  18]), max_size=(180, 240)),
#     Rectangle(p1=[11, 30], p2=[148, 35], color=np.array([100 177  79]), max_size=(180, 240)),
#     Triangle(p1=[80, 63], p2=[113, 125], p3=[143, 130], color=np.array([255 255 255]), max_size=(180, 240)),
#     Triangle(p1=[183, 36], p2=[306, 222], p3=[-100, 212], color=np.array([100 177  79]), max_size=(180, 240)),
#     Ellipse(center=[114, 180], a=284, b=-8, color=np.array([0 0 0]), max_size=(180, 240)),
#     Rectangle(p1=[43, 94], p2=[117, 189], color=np.array([199 192 231]), max_size=(180, 240)),
#     Rectangle(p1=[84, 186], p2=[101, 215], color=np.array([119   1  18]), max_size=(180, 240)),
#     Triangle(p1=[53, 41], p2=[20, -33], p3=[11, 65], color=np.array([231 129  48]), max_size=(180, 240)),
#     Rectangle(p1=[103, 121], p2=[132, 240], color=np.array([196 230  48]), max_size=(180, 240)),
#     Rectangle(p1=[34, 171], p2=[96, 179], color=np.array([128 128 128]), max_size=(180, 240)),
#     Rectangle(p1=[106, 5], p2=[107, 222], color=np.array([196 196 196]), max_size=(180, 240)),
#     Triangle(p1=[20, 117], p2=[-19, 36], p3=[169, 99], color=np.array([255 255 255]), max_size=(180, 240)),
#     Ellipse(center=[77, 119], a=69, b=11, color=np.array([196 196 196]), max_size=(180, 240)),
#     Rectangle(p1=[90, 206], p2=[112, 217], color=np.array([196 196 196]), max_size=(180, 240)),
#     Ellipse(center=[28, 122], a=102, b=12, color=np.array([196 230  48]), max_size=(180, 240)),
#     Rectangle(p1=[69, 80], p2=[81, 206], color=np.array([119   1  18]), max_size=(180, 240)),
#     Rectangle(p1=[85, 70], p2=[137, 115], color=np.array([100 177  79]), max_size=(180, 240)),
#     Rectangle(p1=[81, 72], p2=[133, 214], color=np.array([119   1  18]), max_size=(180, 240)),
#     Rectangle(p1=[64, 84], p2=[103, 202], color=np.array([119   1  18]), max_size=(180, 240)),
#     Rectangle(p1=[44, 97], p2=[84, 190], color=np.array([119   1  18]), max_size=(180, 240)),
#     Rectangle(p1=[0, 0], p2=[7, 240], color=np.array([0 0 0]), max_size=(180, 240)),
#     Rectangle(p1=[91, 72], p2=[139, 213], color=np.array([237 229 179]), max_size=(180, 240)),
#     Rectangle(p1=[33, 172], p2=[63, 185], color=np.array([119   1  18]), max_size=(180, 240)),
#     Ellipse(center=[83, 105], a=9, b=-7, color=np.array([174 218 234]), max_size=(180, 240)),
#     Rectangle(p1=[75, 76], p2=[91, 210], color=np.array([119   1  18]), max_size=(180, 240)),
#     Rectangle(p1=[119, 93], p2=[139, 111], color=np.array([172 123  90]), max_size=(180, 240)),
#     Rectangle(p1=[91, 215], p2=[106, 228], color=np.array([255 255 255]), max_size=(180, 240)),
#     Rectangle(p1=[53, 89], p2=[70, 196], color=np.array([119   1  18]), max_size=(180, 240)),
#     Rectangle(p1=[7, 17], p2=[13, 194], color=np.array([174 218 234]), max_size=(180, 240)),
#     Rectangle(p1=[42, 47], p2=[58, 92], color=np.array([255 255 255]), max_size=(180, 240)),
#     Rectangle(p1=[21, 20], p2=[42, 40], color=np.array([251 241  42]), max_size=(180, 240)),
# ]


demo_pic = image.imread('pic/demo_pic_small.jpg').astype(np.int16)
init(demo_pic)
