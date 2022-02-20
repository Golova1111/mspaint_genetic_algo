import numpy as np
from matplotlib import image

from ga_run import init


demo_pic = image.imread('pic/demo_pic.jpg').astype(np.int16)
init(demo_pic)
