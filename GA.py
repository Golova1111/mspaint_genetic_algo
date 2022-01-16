import random
import time

import numpy as np
from matplotlib import image, pyplot as plt
from Picture import Picture
from Rectangle import Rectangle


class GeneticAlgo:
    POPULATION_SIZE = PS = 250
    EPOCHS = 301
    PART_OF_POPULATION = PoP = 8
    MAX_RESULT_STABLE = 8

    def __init__(self, picture, fignum, prev_winner):
        self.picture = picture
        self.fignum = fignum
        self.prev_winner = prev_winner

        # self.POPULATION_SIZE = 30 * self.fignum

        if not prev_winner:
            self.population = [
                Picture.generate_default(self.picture, max_fignum=self.fignum)
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
            sorted_population = sorted(self.population, key=lambda x: x.score(self.picture))
            best_population = sorted_population[: self.PS // self.PoP]

            best_round_score = best_population[0].delta(self.picture)

            if epoch and epoch % 100 == 0:
                self.visualize(best_population, title=(
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

            if best_round_score < best_score:
                not_changes_in_results = 0
                best_elem = best_population[0]
                best_score = best_round_score

                print(f"Epoch {epoch}: best_score: {best_score}, improved")

            else:
                not_changes_in_results += 1
                print(f"Epoch {epoch}: best_score: {best_score}, remain the same")

            if not_changes_in_results > self.MAX_RESULT_STABLE - 2:
                # if works bad: try to hard-reboot the game
                self.generate_similar(best_elem)

            if not_changes_in_results > self.MAX_RESULT_STABLE and epoch > self.cold_start:
                return best_elem, best_score

            # best-population-approach
            # for num, elem in enumerate(best_population):
            #     self.population[num] = elem
            #
            #     for cp in range(1, self.PART_OF_POPULATION):
            #         if random.random() < 0.5:
            #             curr_result = elem.delta(self.picture)
            #             s = 0
            #             while s < 3:
            #                 self.population[cp * self.PS // self.PoP + num] = Picture.full_mutate(
            #                     elem
            #                 )
            #                 new_result = self.population[cp * self.PS // self.PoP + num].delta(self.picture)
            #
            #                 if new_result < curr_result:
            #                     break
            #                 s += 1
            #             else:
            #                 self.population[cp * self.PS // self.PoP + num] = self.crossover(
            #                     *random.sample(best_population, k=2)
            #                 )
            #         else:
            #             self.population[cp * self.PS // self.PoP + num] = self.crossover(
            #                 *random.sample(best_population, k=2)
            #             )

            for num, elem in enumerate(self.population):
                s = 0
                while s < max(5, self.fignum):
                    mutate_element = Picture.full_mutate(elem)
                    if mutate_element.score(self.picture) < elem.score(self.picture):
                        self.population[num] = mutate_element
                        break
                    s += 1
                else:
                    s = 0
                    while s < max(3, self.fignum // 2):
                        mutate_element = self.crossover(elem, random.choice(best_population))
                        if mutate_element.score(self.picture) < elem.score(self.picture):
                            self.population[num] = mutate_element
                            break
                        s += 1

        return best_elem, best_score

    @staticmethod
    def crossover(elem1, elem2):
        p = Picture(size=elem1.size, max_fignum=elem1.max_fignum)
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
            for _ in range(2 * self.fignum * self.POPULATION_SIZE)
        ]
        self.population[0] = prev_winner
        self.population = sorted(
            self.population, key=lambda x: x.score(self.picture)
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

