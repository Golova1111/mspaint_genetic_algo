from numba import cuda

from GA import GeneticAlgo

d_picture = None


def init(demo_pic):
    figures_number = 1
    max_figures_number = 100
    figures_delta = 1

    d_picture = cuda.to_device(demo_pic)
    prev_winner = None

    # ===============

    # prev_winner = Picture.generate_default(
    #     picture=demo_pic,
    #     max_fignum=figures_number
    # )

    # c = 2

    # prev_winner = Picture(size=(360 * c, 480 * c))
    #
    # prev_winner.parts = [
    #     Rectangle(p1=[ 23 * c,  28 * c], p2=[324 * c, 480 * c], color=np.array([237, 229, 179]), max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[ 17 * c,  99 * c], p2=[181 * c, 480 * c], color=np.array([174, 218, 234]), max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[ 88 * c, 183 * c], p2=[179 * c, 392 * c], color=np.array([119, 1, 18]),    max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[278 * c,   0 * c], p2=[360 * c, 480 * c], color=np.array([100, 177, 79]),  max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[  0 * c,   0 * c], p2=[ 14 * c, 480 * c], color=np.array([0, 0, 0]),       max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[216 * c,   0 * c], p2=[278 * c, 143 * c], color=np.array([196, 230, 48]),  max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[139 * c, 148 * c], p2=[181 * c, 423 * c], color=np.array([119, 1, 18]),    max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[102 * c,  28 * c], p2=[222 * c, 141 * c], color=np.array([255, 255, 255]), max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[269 * c,   0 * c], p2=[331 * c, 143 * c], color=np.array([100, 177, 79]),  max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[347 * c,   0 * c], p2=[360 * c, 480 * c], color=np.array([0, 0, 0]),       max_size=(360 * c, 480 * c)),
    #     Rectangle(p1=[ 33 * c,  25 * c], p2=[101 * c,  98 * c], color=np.array([231, 129, 48]),  max_size=(360 * c, 480 * c)),
    # ]

    # prev_winner.gen_picture()
    # prev_winner.visualize(f"score: {prev_winner.score(demo_pic)}")

    prev_best_score = None
    while figures_number < max_figures_number:
        ga = GeneticAlgo(
            picture=demo_pic,
            d_picture=d_picture,
            fignum=figures_number,
            prev_winner=prev_winner
        )
        prev_winner, best_score = ga.run()

        if prev_best_score:
            if best_score / prev_best_score > 0.9999:
                break

        prev_best_score = best_score

        print(f"figures: {figures_number}, score: {best_score}")

        print("[")
        for elem in prev_winner.parts:
            print(f"    {elem},")
        print("]")

        prev_winner.visualize(title=f"== BEST for {figures_number} figures, score {best_score} == ")
        figures_number += figures_delta


# [
#     Rectangle(p1=[23, 28], p2=[324, 480],   color=np.array([237, 229, 179]), max_size=(240, 320)),
#     Rectangle(p1=[17, 99], p2=[181, 480],   color=np.array([174, 218, 234]), max_size=(240, 320)),
#     Rectangle(p1=[88, 183], p2=[174, 392],  color=np.array([119,   1,  18]), max_size=(240, 320)),
#     Rectangle(p1=[278, 0], p2=[360, 480],   color=np.array([100, 177,  79]), max_size=(240, 320)),
#     Rectangle(p1=[0, 0], p2=[14, 480],      color=np.array([0  , 0  ,  0]), max_size=(240, 320)),
#     Rectangle(p1=[216, 0], p2=[278, 143],   color=np.array([196, 230,  48]), max_size=(240, 320)),
#     Rectangle(p1=[139, 148], p2=[181, 423], color=np.array([119,   1,  18]), max_size=(240, 320)),
#     Rectangle(p1=[102, 28], p2=[222, 141],  color=np.array([255, 255, 255]), max_size=(240, 320)),
#     Rectangle(p1=[267, 0], p2=[318, 138],   color=np.array([100, 177,  79]), max_size=(240, 320)),
#     Rectangle(p1=[16, 232], p2=[79, 232],   color=np.array([255, 255, 255]), max_size=(240, 320)),
#     Rectangle(p1=[345, 41], p2=[360, 419],  color=np.array([0  , 0  , 0]), max_size=(240, 320)),
# ]



# [
#     Rectangle(p1=[22, 32], p2=[340, 429], color=np.array([237 229 179]), max_size=(360, 480)),
#     Rectangle(p1=[209, 426], p2=[285, 480], color=np.array([196 230  48]), max_size=(360, 480)),
#     Rectangle(p1=[239, 186], p2=[310, 222], color=np.array([172 123  90]), max_size=(360, 480)),
#     Rectangle(p1=[17, 16], p2=[159, 480], color=np.array([174 218 234]), max_size=(360, 480)),
#     Rectangle(p1=[88, 176], p2=[156, 391], color=np.array([119   1  18]), max_size=(360, 480)),
#     Rectangle(p1=[278, 0], p2=[360, 480], color=np.array([100 177  79]), max_size=(360, 480)),
#     Rectangle(p1=[0, 0], p2=[14, 480], color=np.array([0 0 0]), max_size=(360, 480)),
#     Rectangle(p1=[214, 0], p2=[278, 143], color=np.array([196 230  48]), max_size=(360, 480)),
#     Rectangle(p1=[135, 151], p2=[181, 431], color=np.array([119   1  18]), max_size=(360, 480)),
#     Rectangle(p1=[109, 30], p2=[221, 140], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[269, 0], p2=[349, 143], color=np.array([100 177  79]), max_size=(360, 480)),
#     Rectangle(p1=[347, 0], p2=[360, 480], color=np.array([0 0 0]), max_size=(360, 480)),
#     Rectangle(p1=[30, 27], p2=[98, 96], color=np.array([231 129  48]), max_size=(360, 480)),
#     Rectangle(p1=[42, 39], p2=[84, 81], color=np.array([251 241  42]), max_size=(360, 480)),
#     Rectangle(p1=[74, 415], p2=[159, 480], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[66, 343], p2=[143, 370], color=np.array([119   1  18]), max_size=(360, 480)),
#     Rectangle(p1=[60, 108], p2=[112, 191], color=np.array([255 255 255]), max_size=(360, 480)),
# ]




# [
#     Rectangle(p1=[22, 57], p2=[351, 429], color=np.array([237 229 179]), max_size=(360, 480)),
#     Rectangle(p1=[209, 426], p2=[285, 480], color=np.array([196 230  48]), max_size=(360, 480)),
#     Rectangle(p1=[239, 186], p2=[303, 222], color=np.array([172 123  90]), max_size=(360, 480)),
#     Rectangle(p1=[17, 15], p2=[127, 480], color=np.array([174 218 234]), max_size=(360, 480)),
#     Rectangle(p1=[88, 180], p2=[151, 394], color=np.array([119   1  18]), max_size=(360, 480)),
#     Rectangle(p1=[278, 0], p2=[360, 480], color=np.array([100 177  79]), max_size=(360, 480)),
#     Rectangle(p1=[197, 151], p2=[224, 183], color=np.array([174 218 234]), max_size=(360, 480)),
#     Rectangle(p1=[214, 0], p2=[269, 143], color=np.array([196 230  48]), max_size=(360, 480)),
#     Rectangle(p1=[127, 132], p2=[181, 430], color=np.array([119   1  18]), max_size=(360, 480)),
#     Rectangle(p1=[109, 31], p2=[221, 140], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[269, 0], p2=[357, 143], color=np.array([100 177  79]), max_size=(360, 480)),
#     Rectangle(p1=[347, 0], p2=[360, 480], color=np.array([0 0 0]), max_size=(360, 480)),
#     Rectangle(p1=[30, 27], p2=[98, 97], color=np.array([231 129  48]), max_size=(360, 480)),
#     Rectangle(p1=[42, 39], p2=[84, 81], color=np.array([251 241  42]), max_size=(360, 480)),
#     Rectangle(p1=[74, 411], p2=[154, 480], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[14, 109], p2=[164, 152], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[60, 157], p2=[107, 193], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[83, 94], p2=[145, 166], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[66, 343], p2=[115, 370], color=np.array([119   1  18]), max_size=(360, 480)),
#     Rectangle(p1=[65, 208], p2=[86, 343], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[236, 89], p2=[254, 106], color=np.array([119   1  18]), max_size=(360, 480)),
#     Rectangle(p1=[264, 426], p2=[286, 480], color=np.array([100 177  79]), max_size=(360, 480)),
#     Rectangle(p1=[48, 382], p2=[106, 415], color=np.array([255 255 255]), max_size=(360, 480)),
#     Rectangle(p1=[0, 0], p2=[14, 480], color=np.array([0 0 0]), max_size=(360, 480)),
# ]
