import numpy as np
from numba import cuda

from Figures.Ellipse import Ellipse
from Figures.Line import Line
from Picture import Picture
from Color import Color, get_similar_color, all_colors_component_list, get_color
from Figures.Rectangle import Rectangle
from Figures.Triangle import Triangle

c = Color()


def test_color_similarity():
    for i in range(100):
        similar_color = get_similar_color(Color.RED)
        print(c.color_dict[tuple(similar_color)])


def test_picture_color_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    col = c.GRAY1

    p.parts.append(
        Rectangle(p1=(100, 100), p2=(200, 200), angle=0, color=col, max_size=(240, 320)),
    )

    for x in range(10):
        for y in range(10):
            fig = Rectangle(p1=(10 * x, 10 * y), p2=(10 * (x + 1), 10 * (y + 1)), angle=0, color=col, max_size=(240, 320))
            fig._color_mutate()
            p.parts.append(fig)

    p.gen_picture()
    p.visualize(is_save=False)


def test_picture_rectangle_generation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    for x in range(10):
        f = Rectangle.gen_random(size=(240, 320))
        f.color = c.RED
        f._color_mutate()

        p.parts.append(f)
        print(f)

    p.gen_picture()
    p.visualize(is_save=False)


def test_picture_line_generation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    for x in range(100):
        f = Line.gen_random(size=(240, 320))
        f.color = c.RED
        f._color_mutate()

        p.parts.append(f)
        print(f)

    p.gen_picture()
    p.visualize(is_save=False)


def test_picture_ellipse_generation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    for x in range(50):
        f = Ellipse.gen_random(size=(240, 320))
        f.color = c.RED
        f._color_mutate()

        p.parts.append(f)
        print(f)

    p.gen_picture()
    p.visualize(is_save=False)


def test_picture_triangle_generation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    for x in range(50):
        f = Triangle.gen_random(size=(240, 320), is_small=3)
        f.color = c.RED
        f._color_mutate()

        p.parts.append(f)
        print(f)

    p.gen_picture()
    p.visualize(is_save=False)


def test_triangle_rectangle_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    p.parts.append(
        Triangle(p1=(100, 100), p2=(200, 200), p3=(50, 200), color=c.LIGHTBLUE, max_size=(240, 320)),
    )

    p.gen_picture()
    p.visualize(is_save=False)

    p.parts[0] = p.parts[0]._rectangle_mutate()
    print(p.parts[0])

    p.gen_picture()
    p.visualize(is_save=False)


def test_ellipse_triangle_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    p.parts.append(
        Ellipse(center=(100, 100), a=60, b=30, angle=1.5, color=c.LIGHTBLUE, max_size=(240, 320)),
    )

    p.gen_picture()
    p.visualize(is_save=False)

    p.parts[0] = p.parts[0]._triangle_mutate()
    print(p.parts[0])

    p.gen_picture()
    p.visualize(is_save=False)


def test_rectangle_triangle_mutation():
    picture = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(240, 320), d_picture=d_picture)

    p.parts.append(
        Rectangle(p1=(50, 50), p2=(120, 200), angle=1, color=c.LIGHTBLUE, max_size=(240, 320)),
    )

    p.gen_picture()
    p.visualize(is_save=False)

    p.parts[0] = p.parts[0]._triangle_mutate()
    print(p.parts[0])

    p.gen_picture()
    p.visualize(is_save=False)


def picture_visualize():
    picture = np.random.randint(low=0, high=255, size=(180, 240, 3), dtype=np.int16)
    d_picture = cuda.to_device(picture)
    prev_winner = Picture(size=(180, 240), d_picture=d_picture, max_fignum=143)
    #
    prev_winner.parts = [
        Rectangle(p1=[240, 74], p2=[180, 77], color=[237, 229, 179], color_delta=-39, angle=1.0396211961160324, max_size=(240, 180)),
        Ellipse(center=[171, 65], a=192, b=126, color=[0, 0, 0], color_delta=39, angle=-2.5712666854047095, max_size=(240, 180)),
        Rectangle(p1=[146, 51], p2=[180, 57], color=[196, 196, 196], color_delta=-16, angle=1.6668765209571832, max_size=(240, 180)),
        Rectangle(p1=[227, 49], p2=[152, 87], color=[196, 196, 196], color_delta=-17, angle=-2.5444778647121726, max_size=(240, 180)),
        Ellipse(center=[97, 1], a=-58, b=83, color=[0, 0, 0], color_delta=42, angle=1.2569608110358303, max_size=(240, 180)),
        Ellipse(center=[65, 126], a=71, b=89, color=[196, 196, 196], color_delta=-28, angle=0.3238302132508983, max_size=(240, 180)),
        Rectangle(p1=[168, 39], p2=[187, 66], color=[0, 0, 0], color_delta=38, angle=0.7933635118228501, max_size=(240, 180)),
        Rectangle(p1=[34, 88], p2=[69, 95], color=[0, 0, 0], color_delta=27, angle=0.9561593597500481, max_size=(240, 180)),
        Rectangle(p1=[33, 2], p2=[16, 75], color=[0, 0, 0], color_delta=45, angle=-1.677689330791232, max_size=(240, 180)),
        Rectangle(p1=[130, 165], p2=[240, 180], color=[196, 196, 196], color_delta=-8, angle=0.0036066406657239945, max_size=(240, 180)),
        Rectangle(p1=[5, 105], p2=[8, 109], color=[196, 196, 196], color_delta=-11, angle=0.10808378380927103, max_size=(240, 180)),
        Triangle(p1=[110, 52], p2=[153, 92], p3=[98, 83], color=[242, 202, 41], color_delta=5, max_size=(240, 180)),
        Rectangle(p1=[100, 30], p2=[106, 49], color=[100, 177, 79], color_delta=-2, angle=-2.948260129569638, max_size=(240, 180)),
        Ellipse(center=[28, 7], a=16, b=-62, color=[128, 128, 128], color_delta=15, angle=2.0237989523450115, max_size=(240, 180)),
        Ellipse(center=[181, 80], a=67, b=39, color=[0, 0, 0], color_delta=25, angle=-3.030488705519693, max_size=(240, 180)),
        Ellipse(center=[66, 63], a=38, b=40, color=[237, 229, 179], color_delta=-18, angle=0.19533937442846092, max_size=(240, 180)),
        Ellipse(center=[26, 82], a=40, b=5, color=[196, 196, 196], color_delta=-23, angle=-2.1861906276851655, max_size=(240, 180)),
        Rectangle(p1=[0, 116], p2=[135, 180], color=[196, 196, 196], color_delta=-33, angle=3.185222500951248, max_size=(240, 180)),
        Rectangle(p1=[223, 44], p2=[240, 80], color=[128, 128, 128], color_delta=19, angle=1.84208076056227, max_size=(240, 180)),
        Rectangle(p1=[132, 76], p2=[177, 83], color=[0, 0, 0], color_delta=34, angle=0.5889044554491895, max_size=(240, 180)),
        Triangle(p1=[110, 146], p2=[135, 156], p3=[134, 169], color=[174, 218, 234], color_delta=-19, max_size=(240, 180)),
        Rectangle(p1=[82, 131], p2=[106, 179], color=[128, 128, 128], color_delta=-22, angle=-1.7761974774216076, max_size=(240, 180)),
        Rectangle(p1=[123, 71], p2=[149, 112], color=[237, 229, 179], color_delta=-15, angle=2.1416017144294006, max_size=(240, 180)),
        Rectangle(p1=[34, 63], p2=[60, 99], color=[172, 123, 90], color_delta=4, angle=2.200959990039231, max_size=(240, 180)),
        Ellipse(center=[178, 25], a=2, b=16, color=[0, 0, 0], color_delta=41, angle=0.36313704944668507, max_size=(240, 180)),
        Rectangle(p1=[7, 106], p2=[42, 122], color=[128, 128, 128], color_delta=-16, angle=-2.2750071552966515, max_size=(240, 180)),
        Rectangle(p1=[127, 108], p2=[145, 129], color=[196, 196, 196], color_delta=-26, angle=-2.2642130978484354, max_size=(240, 180)),
        Triangle(p1=[131, 170], p2=[135, 152], p3=[137, 128], color=[237, 229, 179], color_delta=-28, max_size=(240, 180)),
        Rectangle(p1=[25, 117], p2=[51, 171], color=[128, 128, 128], color_delta=-20, angle=-1.3188296122303347, max_size=(240, 180)),
        Triangle(p1=[20, 169], p2=[15, 177], p3=[3, 178], color=[128, 128, 128], color_delta=-17, max_size=(240, 180)),
        Rectangle(p1=[202, 0], p2=[157, 34], color=[196, 196, 196], color_delta=1, angle=-2.6426867747201497, max_size=(240, 180)),
        Rectangle(p1=[138, 64], p2=[240, 168], color=[0, 0, 0], color_delta=36, angle=0.0381214654252513, max_size=(240, 180)),
        Rectangle(p1=[224, 62], p2=[180, 107], color=[0, 0, 0], color_delta=41, angle=0.13875487016819574, max_size=(240, 180)),
        Rectangle(p1=[75, 20], p2=[57, 58], color=[196, 196, 196], color_delta=-11, angle=0.7324664648620283, max_size=(240, 180)),
        Rectangle(p1=[3, 72], p2=[39, 82], color=[0, 0, 0], color_delta=14, angle=-1.3580307143553052, max_size=(240, 180)),
        Ellipse(center=[54, 34], a=16, b=-20, color=[196, 196, 196], color_delta=-23, angle=1.1105323013146238, max_size=(240, 180)),
        Triangle(p1=[30, 52], p2=[95, 32], p3=[122, 62], color=[238, 176, 202], color_delta=-20, max_size=(240, 180)),
        Triangle(p1=[7, 140], p2=[61, 185], p3=[95, 114], color=[0, 0, 0], color_delta=5, max_size=(240, 180)),
        Rectangle(p1=[0, 65], p2=[81, 88], color=[172, 123, 90], color_delta=8, angle=-3.0189963708457057, max_size=(240, 180)),
        Rectangle(p1=[143, 66], p2=[240, 114], color=[0, 0, 0], color_delta=32, angle=0.008708554921232802, max_size=(240, 180)),
        Rectangle(p1=[37, 31], p2=[58, 45], color=[128, 128, 128], color_delta=-13, angle=-0.9501873479117587, max_size=(240, 180)),
        Rectangle(p1=[85, 65], p2=[101, 94], color=[196, 196, 196], color_delta=-15, angle=-1.528094556263712, max_size=(240, 180)),
        Triangle(p1=[36, 147], p2=[4, 101], p3=[51, 126], color=[128, 128, 128], color_delta=-19, max_size=(240, 180)),
        Rectangle(p1=[71, 90], p2=[75, 180], color=[128, 128, 128], color_delta=-24, angle=-3.216316846284141, max_size=(240, 180)),
        Rectangle(p1=[70, 120], p2=[53, 180], color=[128, 128, 128], color_delta=-17, angle=0.109644592101186, max_size=(240, 180)),
        Ellipse(center=[127, 140], a=26, b=8, color=[196, 196, 196], color_delta=-30, angle=1.5265753163586122, max_size=(240, 180)),
        Rectangle(p1=[91, 61], p2=[110, 71], color=[172, 123, 90], color_delta=3, angle=-0.9016043091755652, max_size=(240, 180)),
        Triangle(p1=[234, 153], p2=[238, 180], p3=[228, 198], color=[196, 196, 196], color_delta=-11, max_size=(240, 180)),
        Rectangle(p1=[137, 75], p2=[143, 112], color=[196, 196, 196], color_delta=-24, angle=3.137966274660108, max_size=(240, 180)),
        Triangle(p1=[146, 71], p2=[156, 89], p3=[156, 57], color=[128, 128, 128], color_delta=18, max_size=(240, 180)),
        Rectangle(p1=[130, 114], p2=[162, 101], color=[0, 0, 0], color_delta=35, angle=-2.3444457755940853, max_size=(240, 180)),
        Triangle(p1=[129, 92], p2=[200, 78], p3=[243, 97], color=[0, 0, 0], color_delta=26, max_size=(240, 180)),
        Triangle(p1=[1, 98], p2=[36, 73], p3=[-3, 21], color=[0, 0, 0], color_delta=36, max_size=(240, 180)),
        Rectangle(p1=[220, 97], p2=[248, 83], color=[196, 196, 196], color_delta=-21, angle=-2.649370605191679, max_size=(240, 180)),
        Rectangle(p1=[107, 98], p2=[82, 133], color=[128, 128, 128], color_delta=-18, angle=-2.4195044091900275, max_size=(240, 180)),
        Rectangle(p1=[28, 32], p2=[35, 39], color=[65, 74, 203], color_delta=34, angle=-2.9185263545797855, max_size=(240, 180)),
        Rectangle(p1=[3, 6], p2=[22, 25], color=[199, 192, 231], color_delta=-2, angle=-0.1472408326397423, max_size=(240, 180)),
        Triangle(p1=[88, 31], p2=[83, 55], p3=[127, 25], color=[0, 0, 0], color_delta=26, max_size=(240, 180)),
        Triangle(p1=[-33, 26], p2=[23, 29], p3=[8, 42], color=[0, 0, 0], color_delta=42, max_size=(240, 180)),
        Rectangle(p1=[205, 67], p2=[178, 88], color=[100, 177, 79], color_delta=25, angle=-1.5072522991768063, max_size=(240, 180)),
        Rectangle(p1=[83, 63], p2=[125, 99], color=[172, 123, 90], color_delta=-7, angle=3.194399708170882, max_size=(240, 180)),
        Rectangle(p1=[161, 80], p2=[196, 91], color=[0, 0, 0], color_delta=40, angle=0.14954034267408314, max_size=(240, 180)),
        Ellipse(center=[214, 81], a=53, b=13, color=[196, 196, 196], color_delta=-18, angle=-0.16677808876495892, max_size=(240, 180)),
        Triangle(p1=[196, 82], p2=[184, 98], p3=[191, 93], color=[196, 196, 196], color_delta=-28, max_size=(240, 180)),
        Triangle(p1=[2, 50], p2=[3, 77], p3=[-56, 78], color=[196, 196, 196], color_delta=-16, max_size=(240, 180)),
        Triangle(p1=[16, 79], p2=[-19, 61], p3=[18, 73], color=[196, 196, 196], color_delta=-4, max_size=(240, 180)),
        Ellipse(center=[149, 165], a=-16, b=-7, color=[237, 229, 179], color_delta=-9, angle=-0.6385121264179091, max_size=(240, 180)),
        Rectangle(p1=[123, 77], p2=[136, 62], color=[237, 229, 179], color_delta=-14, angle=2.588838493323413, max_size=(240, 180)),
        Ellipse(center=[214, 161], a=-19, b=74, color=[0, 0, 0], color_delta=32, angle=-1.9589037455581622, max_size=(240, 180)),
        Triangle(p1=[-8, 107], p2=[28, 82], p3=[-48, 80], color=[196, 196, 196], color_delta=-24, max_size=(240, 180)),
        Triangle(p1=[174, 87], p2=[150, 56], p3=[185, 83], color=[237, 229, 179], color_delta=-12, max_size=(240, 180)),
        Rectangle(p1=[47, 32], p2=[85, 64], color=[172, 123, 90], color_delta=-8, angle=2.935268944952844, max_size=(240, 180)),
        Rectangle(p1=[45, 41], p2=[76, 97], color=[172, 123, 90], color_delta=-1, angle=-0.07273632968613361, max_size=(240, 180)),
        Ellipse(center=[75, 45], a=11, b=-7, color=[172, 123, 90], color_delta=3, angle=1.941493951286773, max_size=(240, 180)),
        Rectangle(p1=[82, 85], p2=[97, 95], color=[128, 128, 128], color_delta=-18, angle=-2.29224177568162, max_size=(240, 180)),
        Rectangle(p1=[147, 83], p2=[157, 91], color=[196, 196, 196], color_delta=-20, angle=-2.875825257730429, max_size=(240, 180)),
        Rectangle(p1=[188, 41], p2=[213, 93], color=[0, 0, 0], color_delta=27, angle=-0.07331194446059804, max_size=(240, 180)),
        Triangle(p1=[221, 54], p2=[122, 56], p3=[141, 65], color=[237, 229, 179], color_delta=-14, max_size=(240, 180)),
        Rectangle(p1=[219, 73], p2=[227, 94], color=[196, 196, 196], color_delta=-27, angle=-0.027724613017443736, max_size=(240, 180)),
        Ellipse(center=[61, 57], a=44, b=-8, color=[172, 123, 90], color_delta=4, angle=3.0984521496649706, max_size=(240, 180)),
        Ellipse(center=[31, 184], a=-16, b=17, color=[128, 128, 128], color_delta=-18, angle=0.8079223231357396, max_size=(240, 180)),
        Triangle(p1=[91, 35], p2=[69, 19], p3=[38, 27], color=[128, 128, 128], color_delta=-8, max_size=(240, 180)),
        Rectangle(p1=[188, 77], p2=[181, 89], color=[237, 229, 179], color_delta=-26, angle=-0.1254389792218513, max_size=(240, 180)),
        Ellipse(center=[151, 99], a=-8, b=6, color=[0, 0, 0], color_delta=34, angle=-0.7733813804784725, max_size=(240, 180)),
        Triangle(p1=[120, 67], p2=[111, -17], p3=[131, 70], color=[0, 0, 0], color_delta=43, max_size=(240, 180)),
        Rectangle(p1=[109, 57], p2=[122, 76], color=[128, 128, 128], color_delta=-11, angle=2.7826724219368284, max_size=(240, 180)),
        Rectangle(p1=[8, 56], p2=[37, 70], color=[0, 0, 0], color_delta=40, angle=3.134289144059111, max_size=(240, 180)),
        Triangle(p1=[195, 110], p2=[198, 56], p3=[210, 57], color=[0, 0, 0], color_delta=17, max_size=(240, 180)),
        Rectangle(p1=[177, 96], p2=[173, 102], color=[174, 218, 234], color_delta=-30, angle=-0.34613762690071176, max_size=(240, 180)),
        Rectangle(p1=[46, 109], p2=[69, 127], color=[128, 128, 128], color_delta=-15, angle=0.8905880939037065, max_size=(240, 180)),
        Rectangle(p1=[98, 13], p2=[157, 44], color=[0, 0, 0], color_delta=41, angle=0.05783755253687837, max_size=(240, 180)),
        Rectangle(p1=[240, 92], p2=[176, 93], color=[0, 0, 0], color_delta=34, angle=1.5509540461127052, max_size=(240, 180)),
        Rectangle(p1=[240, 113], p2=[171, 168], color=[128, 128, 128], color_delta=9, angle=-3.076378563911851, max_size=(240, 180)),
        Rectangle(p1=[240, 157], p2=[219, 163], color=[0, 0, 0], color_delta=32, angle=0.5118491310977736, max_size=(240, 180)),
        Triangle(p1=[15, 63], p2=[-35, 55], p3=[-15, 49], color=[0, 0, 0], color_delta=34, max_size=(240, 180)),
        Rectangle(p1=[84, 91], p2=[71, 108], color=[237, 229, 179], color_delta=-21, angle=-0.12874480970163638, max_size=(240, 180)),
        Rectangle(p1=[52, 54], p2=[95, 46], color=[172, 123, 90], color_delta=-7, angle=-2.889962232634481, max_size=(240, 180)),
        Triangle(p1=[87, 61], p2=[79, 62], p3=[92, 30], color=[0, 0, 0], color_delta=38, max_size=(240, 180)),
        Rectangle(p1=[84, 14], p2=[194, 37], color=[128, 128, 128], color_delta=15, angle=-0.18157291981788093, max_size=(240, 180)),
        Ellipse(center=[209, 14], a=40, b=5, color=[128, 128, 128], color_delta=17, angle=-0.18761691055198806, max_size=(240, 180)),
        Triangle(p1=[70, 5], p2=[129, 10], p3=[161, -62], color=[128, 128, 128], color_delta=18, max_size=(240, 180)),
        Rectangle(p1=[130, 59], p2=[133, 76], color=[237, 229, 179], color_delta=-17, angle=0.6531979666871575, max_size=(240, 180)),
        Triangle(p1=[21, 114], p2=[7, 161], p3=[30, 125], color=[196, 196, 196], color_delta=-36, max_size=(240, 180)),
        Ellipse(center=[87, 60], a=9, b=3, color=[238, 176, 202], color_delta=-26, angle=-1.3155747701963767, max_size=(240, 180)),
        Rectangle(p1=[176, 94], p2=[193, 110], color=[196, 196, 196], color_delta=-7, angle=1.7602539026515147, max_size=(240, 180)),
        Triangle(p1=[13, 163], p2=[-7, 145], p3=[0, 191], color=[128, 128, 128], color_delta=-20, max_size=(240, 180)),
        Rectangle(p1=[123, 114], p2=[116, 163], color=[196, 196, 196], color_delta=-33, angle=3.099822335652175, max_size=(240, 180)),
        Rectangle(p1=[152, 92], p2=[155, 98], color=[0, 0, 0], color_delta=45, angle=1.3702552397109597, max_size=(240, 180)),
        Rectangle(p1=[145, 66], p2=[136, 72], color=[172, 123, 90], color_delta=29, angle=-1.543581568086987, max_size=(240, 180)),
        Ellipse(center=[228, 102], a=-5, b=70, color=[128, 128, 128], color_delta=5, angle=-1.575901842135717, max_size=(240, 180)),
        Triangle(p1=[80, 94], p2=[63, 92], p3=[72, 77], color=[196, 196, 196], color_delta=-10, max_size=(240, 180)),
        Rectangle(p1=[108, 154], p2=[115, 160], color=[0, 0, 0], color_delta=2, angle=-1.9984567206379031, max_size=(240, 180)),
        Rectangle(p1=[187, 121], p2=[184, 108], color=[128, 128, 128], color_delta=7, angle=-0.9121092551441425, max_size=(240, 180)),
        Ellipse(center=[167, 74], a=-12, b=7, color=[172, 123, 90], color_delta=25, angle=-1.5978605200040186, max_size=(240, 180)),
        Ellipse(center=[175, 128], a=-11, b=23, color=[128, 128, 128], color_delta=10, angle=-1.1698518299698868, max_size=(240, 180)),
        Triangle(p1=[72, 19], p2=[57, 15], p3=[36, 25], color=[196, 196, 196], color_delta=-12, max_size=(240, 180)),
        Rectangle(p1=[153, 122], p2=[179, 107], color=[196, 196, 196], color_delta=-5, angle=0.3465608387114208, max_size=(240, 180)),
        Rectangle(p1=[0, 157], p2=[22, 164], color=[0, 0, 0], color_delta=2, angle=1.8616158614688776, max_size=(240, 180)),
        Ellipse(center=[107, 136], a=6, b=22, color=[128, 128, 128], color_delta=-20, angle=-2.836153894329851, max_size=(240, 180)),
        Rectangle(p1=[43, 121], p2=[77, 114], color=[0, 0, 0], color_delta=9, angle=0.8826767183971652, max_size=(240, 180)),
        Rectangle(p1=[133, 178], p2=[277, 231], color=[0, 0, 0], color_delta=23, angle=0, max_size=(240, 180)),
        Triangle(p1=[83, 112], p2=[79, 79], p3=[92, 46], color=[237, 229, 179], color_delta=-28, max_size=(240, 180)),
        Rectangle(p1=[117, 176], p2=[151, 180], color=[128, 128, 128], color_delta=-19, angle=-1.711104764724269, max_size=(240, 180)),
        Ellipse(center=[114, 55], a=-9, b=3, color=[237, 229, 179], color_delta=-23, angle=1.001971147894505, max_size=(240, 180)),
        Triangle(p1=[215, 48], p2=[221, 61], p3=[249, 58], color=[0, 0, 0], color_delta=43, max_size=(240, 180)),
        Rectangle(p1=[62, 73], p2=[90, 73], color=[196, 196, 196], color_delta=-39, angle=-2.547213869713517, max_size=(240, 180)),
        Rectangle(p1=[133, 57], p2=[141, 43], color=[196, 196, 196], color_delta=-11, angle=1.5077913077598901, max_size=(240, 180)),
        Triangle(p1=[68, 136], p2=[68, 175], p3=[72, 181], color=[196, 196, 196], color_delta=-32, max_size=(240, 180)),
        Rectangle(p1=[66, 84], p2=[77, 87], color=[172, 123, 90], color_delta=18, angle=1.6525482888143845, max_size=(240, 180)),
        Ellipse(center=[124, 96], a=-5, b=-3, color=[172, 123, 90], color_delta=-1, angle=-0.20049748466207298, max_size=(240, 180)),
        Rectangle(p1=[33, 121], p2=[49, 133], color=[128, 128, 128], color_delta=-17, angle=1.6382471868822015, max_size=(240, 180)),
        Rectangle(p1=[110, 163], p2=[94, 180], color=[128, 128, 128], color_delta=-20, angle=-1.4644396861983393, max_size=(240, 180)),
        Rectangle(p1=[72, 25], p2=[69, 20], color=[238, 176, 202], color_delta=-16, angle=-1.1877421647025896, max_size=(240, 180)),
        Rectangle(p1=[161, 31], p2=[165, 91], color=[0, 0, 0], color_delta=16, angle=-1.6256127844646708, max_size=(240, 180)),
        Rectangle(p1=[114, 99], p2=[132, 103], color=[237, 229, 179], color_delta=-22, angle=0, max_size=(240, 180)),
        Ellipse(center=[198, 66], a=5, b=5, color=[0, 0, 0], color_delta=39, angle=1.3766696700939463, max_size=(240, 180)),
        Rectangle(p1=[176, 63], p2=[180, 83], color=[237, 229, 179], color_delta=-18, angle=-0.17252347000623175, max_size=(240, 180)),
        Rectangle(p1=[107, 5], p2=[127, 0], color=[0, 0, 0], color_delta=45, angle=0.028295289384688713, max_size=(240, 180)),
        Rectangle(p1=[205, 154], p2=[169, 162], color=[128, 128, 128], color_delta=8, angle=-2.721153801350282, max_size=(240, 180)),
        Rectangle(p1=[180, 54], p2=[176, 90], color=[237, 229, 179], color_delta=-15, angle=-3.133266395923573, max_size=(240, 180)),
        Rectangle(p1=[100, 30], p2=[95, 48], color=[128, 128, 128], color_delta=18, angle=-1.2456199027317214, max_size=(240, 180)),
        Triangle(p1=[33, 66], p2=[6, 66], p3=[32, 59], color=[196, 196, 196], color_delta=-9, max_size=(240, 180)),
    ]
    #
    prev_winner.gen_picture()
    prev_winner.visualize(is_save=False)


def color_test():
    picture = np.zeros(shape=(220, 600, 3), dtype=np.float)
    d_picture = cuda.to_device(picture)
    p = Picture(size=(220, 600), d_picture=d_picture)

    i = 0
    cy = cx = 0
    cy = -1
    curr_color = None

    try:
        while True:
        # for y in range(0, 32):
        #     for x in range(0, 32):
                color, color_delta = all_colors_component_list[i]
                if color != curr_color:
                    curr_color = color
                    cy += 1
                    cx = 0

                print(color, color_delta, "-->", get_color(color, color_delta) , "-->", cx, cy)

                f = Rectangle.gen_random(size=(220, 600))
                f.p1 = [10 * cx, 10 * cy]
                f.p2 = [10 * (cx + 1), 10 * (cy + 1)]
                f.angle = 0

                f.color = color
                f.color_delta = color_delta
                f._repr_color = get_color(f.color, f.color_delta)

                p.parts.append(f)
                cx += 1
                # print(f)

                i += 1
    except Exception as ex:
        print(ex)

    p.gen_picture()
    p.visualize(is_save=False)


def main():
    # test_picture_triangle_generation()
    # test_triangle_rectangle_mutation()
    # test_ellipse_triangle_mutation()
    # test_rectangle_triangle_mutation()
    # color_test()
    # test_picture_color_mutation()
    # test_picture_line_generation()
    picture_visualize()


if __name__ == '__main__':
    main()
