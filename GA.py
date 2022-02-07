import pickle
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

        c = 5

        if self.fignum < 5:
            self.stop_increase = 0.996
            self.POPULATION_SIZE = self.PS = 50 * c
        elif self.fignum < 12:
            self.stop_increase = 0.997
            self.POPULATION_SIZE = self.PS = 100 * c
        elif self.fignum < 18:
            self.stop_increase = 0.998
            self.POPULATION_SIZE = self.PS = 150 * c
        else:
            self.stop_increase = 0.9993
            self.POPULATION_SIZE = self.PS = 150 * c

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
            start_time = time.time()
            self.generate_similar()
            end_time = time.time()
            print("generate_similar:", end_time - start_time)

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

            new_population = []

            for num, elem in enumerate(sorted_population[:self.PS // 3]):
                new_population.append(elem)
                new_population.append(Picture.full_mutate(elem))
                new_population.append(Picture.full_mutate(elem))

            for i in range(self.PS // 5):
                elem1, elem2 = random.sample(best_population, k=2)
                if random.random() < 0.2:
                    new_elem1 = self.combine_crossover(elem1, elem2)
                    new_population.append(new_elem1)
                else:  # if random.random() < 0.4:
                    new_elem1, new_elem2 = self.swap_last_layer_crossover(elem1, elem2)
                    new_population.append(new_elem1)
                    new_population.append(new_elem2)
                # else:
                #     new_elem1, new_elem2 = self.append_last_layer_crossover(elem1, elem2)
                #     new_population.append(new_elem1)
                #     new_population.append(new_elem2)

            self.population = sorted(new_population, key=lambda x: x.score(self.d_picture))[: self.PS]

        return best_elem, best_score

    @staticmethod
    def combine_crossover(elem1, elem2):
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

    @staticmethod
    def swap_last_layer_crossover(elem1, elem2):
        p1 = Picture(size=elem1.size, d_picture=elem2.d_picture, max_fignum=elem1.max_fignum)
        p2 = Picture(size=elem1.size, d_picture=elem2.d_picture, max_fignum=elem1.max_fignum)

        p1.parts = elem1.parts[:-1] + [elem2.parts[-1]]
        p2.parts = elem2.parts[:-1] + [elem1.parts[-1]]

        p1.gen_picture()
        p2.gen_picture()
        return p1, p2

    @staticmethod
    def append_last_layer_crossover(elem1, elem2):
        p1 = Picture(size=elem1.size, d_picture=elem2.d_picture, max_fignum=elem1.max_fignum)
        p2 = Picture(size=elem1.size, d_picture=elem2.d_picture, max_fignum=elem1.max_fignum)

        p1.parts = elem1.parts + [elem2.parts[-1]]
        p2.parts = elem2.parts + [elem1.parts[-1]]

        p1.gen_picture()
        p2.gen_picture()
        return p1, p2

    def generate_similar(self):
        self.population = []
        curr_winner = self.prev_winner

        for i in range(10):
            population_part = [
                Picture.generate_similar(curr_winner, max_fignum=self.fignum)
                for _ in range(3000)
            ]
            self.population += population_part

            self.population = sorted(
                self.population, key=lambda x: x.score(self.d_picture)
            )

            print("curr best:", [x.score(self.d_picture) for x in self.population[:5]])

            if self.fignum < 5:
                c = 96 / 100
            elif self.fignum < 10:
                c = 98.5 / 100
            elif self.fignum < 18:
                c = 99.4 / 100
            elif self.fignum < 24:
                c = 99.6 / 100
            else:
                c = 99.85 / 100

            if self.population[0].score(self.d_picture) / self.prev_winner.score(self.d_picture) < c:
                print("break with coeff: ", self.population[0].score(self.d_picture) / self.prev_winner.score(self.d_picture) * 100)
                break

            if curr_winner.score(self.d_picture) > self.population[0].score(self.d_picture):
                print(self.population[0].score(self.d_picture), " --> ", curr_winner.score(self.d_picture))
                curr_winner = self.population[0]

        if curr_winner.score(self.d_picture) < self.population[0].score(self.d_picture):
            print("not improve, ", curr_winner.score(self.d_picture), " --> ", self.population[0].score(self.d_picture))
            self.population[0] = curr_winner

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

