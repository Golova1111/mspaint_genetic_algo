import numpy as np
from matplotlib import image

from ga_test import init

demo_pic = image.imread('pic/berlin_xsm.jpg').astype(np.int16)
init(demo_pic)
