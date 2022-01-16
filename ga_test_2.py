import numpy as np
from matplotlib import image
from numba import cuda

from GA import GeneticAlgo
from Picture import Picture
from Rectangle import Rectangle


def main():
    figures_number = 13
    max_figures_number = 25
    figures_delta = 1

    prev_winner = None

    # ===============

    demo_pic = image.imread('pic/demo_pic_small.jpg').astype(np.int16)
    d_picture = cuda.to_device(demo_pic)

    # prev_winner.gen_picture()
    # prev_winner.visualize(f"score: {prev_winner.score(demo_pic)}")

    while figures_number < max_figures_number:
        ga = GeneticAlgo(
            picture=demo_pic,
            d_picture=d_picture,
            fignum=figures_number,
            prev_winner=prev_winner
        )
        prev_winner, best_score = ga.run()

        print(f"figures: {figures_number}, score: {best_score}")

        print("[")
        for elem in prev_winner.parts:
            print(f"    Rectangle(p1={elem.p1}, p2={elem.p2}, color=np.array({elem.color}), max_size=(360, 480)),")
        print("]")

        prev_winner.visualize(title=f"== BEST for {figures_number} figures, score {best_score} == ")
        figures_number += figures_delta


if __name__ == '__main__':
    main()
