import random

import numpy as np
from matplotlib import image, pyplot as plt
from Picture import Picture


class GeneticAlgo:
    POPULATION_SIZE = PS = 1200
    EPOCHS = 301
    PART_OF_POPULATION = PoP = 8

    def __init__(self, picture):
        self.picture = picture
        self.population = [
            Picture.generate_default(self.picture) for _ in range(self.POPULATION_SIZE)
        ]

    def run(self):
        best_elem = None
        best_score = np.inf

        for i in range(self.EPOCHS):
            sorted_population = sorted(self.population, key=lambda x: x.delta(self.picture))
            best_population = sorted_population[: self.PS // self.PoP]

            best_round_score = best_population[0].delta(self.picture)

            if i % 5 == 0:
                self.visualize(best_population, title=(
                    f"Epoch {i}, "
                    f"delta: {best_round_score}, "
                    f"fig: {len(best_population[0].parts)}"
                ))

                # best_population[0].visualize(
                #     title=(
                #         f"Epoch {i}, "
                #         f"delta: {best_round_score}, "
                #         f"fig: {len(best_population[0].parts)}"
                #     )
                # )

            if best_round_score < best_score:
                best_elem = best_population[0]
                best_score = best_round_score

            for num, elem in enumerate(best_population):
                self.population[num] = elem

                for cp in range(1, self.PART_OF_POPULATION):
                    if random.random() < 0.5:
                        self.population[cp * self.PS // self.PoP + num] = Picture.full_mutate(
                            elem
                        )
                    else:
                        self.population[cp * self.PS // self.PoP + num] = self.crossover(
                            random.choice(best_population), random.choice(best_population)
                        )

        return best_elem, best_score

    @staticmethod
    def crossover(elem1, elem2):
        p = Picture(size=elem1.size)
        min_parts_len = min(
            len(elem1.parts),
            len(elem2.parts)
        )

        for i in range(min_parts_len):
            if random.random() < 0.5:
                p.parts.append(elem1.parts[i])
            else:
                p.parts.append(elem2.parts[i])

        p.gen_picture()
        return p

    def visualize(self, best_population, title=None):
        fig = plt.figure(figsize=(10, 4))

        for i in range(10):
            ax = fig.add_subplot(2, 5, i + 1)
            plt.imshow(best_population[i].picture, interpolation='nearest')
            ax.set_title(f"#{i}")

        if title:
            fig.suptitle(title)

        plt.show()


def main():
    demo_pic = image.imread('pic/demo_pic.jpg')

    ga = GeneticAlgo(demo_pic)
    best_elem, best_score = ga.run()

    print(best_score)


if __name__ == '__main__':
    main()
