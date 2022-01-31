import random
import time

import numpy as np

from matplotlib import image, pyplot as plt
from Picture import Picture


class GeneticAlgo:
    POPULATION_SIZE = PS = 150
    EPOCHS = 301
    PART_OF_POPULATION = PoP = 3
    MAX_RESULT_STABLE = 6

    def __init__(self, picture, d_picture, fignum, prev_winner):
        self.picture = picture
        self.d_picture = d_picture
        self.fignum = fignum
        self.prev_winner = prev_winner
        self.results = []

        if self.fignum < 5:
            self.stop_increase = 0.987
            self.POPULATION_SIZE = self.PS = 50
        elif self.fignum < 12:
            self.stop_increase = 0.993
            self.POPULATION_SIZE = self.PS = 100
        elif self.fignum < 18:
            self.stop_increase = 0.998
            self.POPULATION_SIZE = self.PS = 150
        else:
            self.stop_increase = 0.999
            self.POPULATION_SIZE = self.PS = 150

        # self.POPULATION_SIZE = 30 * self.fignum

        if not prev_winner:
            self.population = [
                Picture.generate_default(
                    self.picture,
                    d_picture=self.d_picture,
                    max_fignum=self.fignum
                )
                for _ in range(self.POPULATION_SIZE)
            ]
            self.cold_start = 0
        else:
            self.generate_similar(prev_winner)
            self.cold_start = 20


    def run(self):
        start_time = time.time()
        best_elem = None
        best_score = np.inf

        not_changes_in_results = 0

        for epoch in range(self.EPOCHS):
            # score = self.population_result()
            sorted_population = sorted(self.population, key=lambda x: x.score(self.d_picture))
            best_population = sorted_population[: self.PS // self.PoP]

            best_round_score = best_population[0].score(self.d_picture)
            self.results.append(best_round_score)

            if epoch and epoch % 100 == 0:
                self.visualize(self.population, title=(
                    f"Epoch {epoch}, "
                    f"delta: {best_round_score}, "
                    f"fig: {len(best_population[0].parts)}, "
                    f"time {time.time() - start_time}"
                ))

                # best_population[0].visualize(
                #     title=(
                #         f"Epoch {i}, "
                #         f"delta: {best_round_score}, "
                #         f"fig: {len(best_population[0].parts)}"
                #     )
                # )

            if epoch > 10:
                delta = self.results[epoch] / self.results[epoch - 10]
            else:
                delta = 0

            if best_round_score < best_score:
                not_changes_in_results = 0
                best_elem = best_population[0]
                best_score = best_round_score

                print(f"Epoch {epoch}: best_score: {best_score}, improved, time: {time.time() - start_time}, delta {delta}")

            else:
                not_changes_in_results += 1
                print(f"Epoch {epoch}: best_score: {best_score}, remain the same, time: {time.time() - start_time}, delta {delta}")

            if not_changes_in_results > self.MAX_RESULT_STABLE and epoch > self.cold_start:
                return best_elem, best_score

            if epoch > self.cold_start and delta > self.stop_increase:
                return best_elem, best_score

            for num, elem in enumerate(self.population):
                s = 0
                while s < max(10, self.fignum):
                    mutate_element = Picture.full_mutate(elem)
                    if mutate_element.score(self.d_picture) < elem.score(self.d_picture):
                        self.population[num] = mutate_element
                        break
                    s += 1
                else:
                    s = 0
                    while s < max(3, self.fignum // 2):
                        mutate_element = self.crossover(elem, random.choice(best_population))
                        if mutate_element.score(self.d_picture) < elem.score(self.d_picture):
                            self.population[num] = mutate_element
                            break
                        s += 1

        return best_elem, best_score

    @staticmethod
    def crossover(elem1, elem2):
        p = Picture(size=elem1.size, d_picture=elem2.d_picture, max_fignum=elem1.max_fignum)
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

    def generate_similar(self, prev_winner):
        self.population = [
            Picture.generate_similar(self.prev_winner, max_fignum=self.fignum)
            for _ in range(
                max(5 * self.fignum * self.POPULATION_SIZE, 12000)
            )
        ]
        self.population[0] = prev_winner
        self.population = sorted(
            self.population, key=lambda x: x.score(self.d_picture)
        )[:self.POPULATION_SIZE]

    def visualize(self, best_population, title=None):
        fig = plt.figure(figsize=(10, 4))

        for i in range(10):
            ax = fig.add_subplot(2, 5, i + 1)
            plt.imshow(best_population[i].picture, interpolation='nearest')
            ax.set_title(f"#{i}")

        if title:
            fig.suptitle(title)

        plt.show()

