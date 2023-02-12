import time

import numpy as np
import matplotlib.pyplot as plt

from LinReg import LinReg

linreg = LinReg()
data = np.loadtxt("14317816.txt", delimiter=",", encoding="UTF-8", dtype=float)


class Candidate:
    def __init__(self, bitstr_size: int, task: str) -> None:
        # inverse binary repr
        self.bitstring: np.ndarray[int] = np.random.randint(0, 2, bitstr_size)
        self.evaluate(task=task)

    def get_solution(self) -> int:
        """Gives the integer representation of the bitstring

        Returns:
            int: Integer to pass to sine function
        """
        num = 0
        for idx, bit_num in enumerate(self.bitstring):
            num += 2**idx * bit_num
        return num / (2 ** (len(self.bitstring) - 7))

    def evaluate(self, task: str) -> float:
        # set fitness attribute
        if task == "a":
            fitness = np.sin(self.get_solution())
        # f)
        elif task == "f":
            fitness = np.sin(self.get_solution())
            if not 5 <= self.get_solution() <= 10:
                fitness -= 1

        # g)
        elif task == "g":
            columns = linreg.get_columns(X=data, bitstring=self.bitstring)
            fitness = linreg.get_fitness(x=columns, y=data[:, -1])
        self.fitness = fitness

    def mutate(
        self,
        genome_mutate_chance: float,
        min_mutate_num: int,
        max_mutate_num: int,
    ) -> None:
        # number of genomes to pick mutate
        num_genomes_to_mutate = np.random.randint(min_mutate_num, max_mutate_num + 1)
        # select random places in bitstring
        genome_indices = np.random.choice(
            np.arange(len(self.bitstring)), num_genomes_to_mutate, replace=False
        )
        # invert bitstring in mutated spots
        for idx in genome_indices:
            if np.random.uniform() < genome_mutate_chance:
                self.bitstring[idx] = 1 - self.bitstring[idx]


def create_population(size: int, bitstr_size: int, task: str) -> list[Candidate]:
    return [Candidate(bitstr_size=bitstr_size, task=task) for _ in range(size)]


def parent_selection(
    population: list[Candidate], num_parents: int, task: str
) -> list[Candidate]:
    if num_parents % 2 != 0:
        raise ValueError("num_parents not dividable by 2")

    # sort population
    if task == "g":
        # get the worst
        population.sort(key=lambda x: x.fitness)
    else:
        # get the best
        population.sort(key=lambda x: x.fitness, reverse=True)
    # return the two first
    return population[:num_parents]


def crossover(
    parents: np.ndarray[Candidate],
    bitstr_size: int,
    task: str,
    num_children: int,
    deterministic: bool,
    ranked: bool,
) -> np.ndarray[Candidate]:

    children = np.array([])

    if deterministic and num_children > len(parents):
        raise ValueError(
            "Cannot choose determinisitc parents to create more children than parents"
        )
    elif deterministic:
        for idx in range(int(len(parents) / 2)):
            pair = np.array([parents[idx * 2], parents[idx * 2 + 1]])
            children = np.append(
                children,
                mate(pair=pair, mutate_chance=0.5, bitstr_size=bitstr_size, task=task),
            )
    else:
        for _ in range(int(num_children / 2)):
            if ranked:
                prob = np.array([x.fitness + 1 for x in parents])
                prob = prob / np.sum(prob)
                pair = np.random.choice(parents, 2, replace=False, p=prob)
            else:
                pair = np.random.choice(parents, 2, replace=False)
            children = np.append(
                children,
                mate(pair=pair, mutate_chance=0.5, bitstr_size=bitstr_size, task=task),
            )

    return children


def mate(
    pair: np.ndarray[Candidate], mutate_chance: float, bitstr_size: int, task: str
) -> np.ndarray[Candidate]:
    # find point to split where to get dna from parents
    split_index = np.random.randint(1, bitstr_size - 1)
    # create child1 and insert dna
    child1 = Candidate(bitstr_size=bitstr_size, task=task)
    child1.bitstring = np.append(
        pair[0].bitstring[:split_index], pair[1].bitstring[split_index:]
    )
    # create child2 and insert dna
    child2 = Candidate(bitstr_size=bitstr_size, task=task)
    child2.bitstring = np.append(
        pair[1].bitstring[:split_index], pair[0].bitstring[split_index:]
    )
    # mutate children
    if np.random.uniform() < mutate_chance:
        child1.mutate(
            genome_mutate_chance=0.5, min_mutate_num=1, max_mutate_num=bitstr_size
        )
    if np.random.uniform() < mutate_chance:
        child2.mutate(
            genome_mutate_chance=0.5, min_mutate_num=1, max_mutate_num=bitstr_size
        )

    child1.evaluate(task=task)
    child2.evaluate(task=task)

    children = np.array([child1, child2])

    return children


def tournament(
    population: list[Candidate], pop_size: int, task: str
) -> list[Candidate]:
    # sort population
    if task == "g":
        # the worst survive
        population.sort(key=lambda x: x.fitness)
    else:
        # the best survive
        population.sort(key=lambda x: x.fitness, reverse=True)

    # return survivors
    return population[:pop_size]


def crowding_tournament(
    parents: list[Candidate], children: list[Candidate], task: str
) -> list[Candidate]:
    winners = np.array([])
    for idx in range(int(len(parents) / 2)):
        # check parent and children pairs for similarity
        p1 = parents[idx * 2]
        p2 = parents[idx * 2 + 1]
        c1 = children[idx * 2]
        c2 = children[idx * 2 + 1]

        # get least diff child to parent 1
        diff_p1_c1 = np.sum(p1.bitstring - c1.bitstring)
        diff_p1_c2 = np.sum(p1.bitstring - c2.bitstring)
        pair_idx = int(np.argmin([diff_p1_c1, diff_p1_c2]))
        # insert the winner from duels into winner list
        winners = np.append(winners, duel(p1, children[idx * 2 + pair_idx], task=task))
        winners = np.append(
            winners, duel(p2, children[idx * 2 + (1 - pair_idx)], task=task)
        )

    return winners.tolist()


def duel(can1: Candidate, can2: Candidate, task: str) -> Candidate:
    if can1.fitness > can2.fitness:
        if task == "g":
            return can2
        return can1
    if task == "g":
        return can1
    return can2


def plot(task: str, name: str, num_ep: int, avg_fit: list[float], pop: list[Candidate]):
    plt.clf()
    if task == "g":
        plt.plot(np.arange(num_ep), avg_fit, label="RMSE feature selected")
        plt.plot(
            np.arange(num_ep),
            np.ones(num_ep) * linreg.get_fitness(data[:, :101], data[:, -1]),
            c="r",
            label="base RMSE",
        )
        # plt.plot(np.arange(num_ep), np.ones(num_ep) * 0.124, c="g", label="0.124")
        plt.legend()
    else:
        plt.plot(np.arange(0, 128, 0.5), np.sin(np.arange(0, 128, 0.5)))
        for i in pop:
            plt.scatter(i.get_solution(), np.sin(i.get_solution()), c="r")
    plt.savefig(name)


def entropy(pop: list[Candidate], bitstr_size: int):
    bitstr_numbers = np.zeros(bitstr_size)
    # adds together all genomes for every candidate
    for x in pop:
        bitstr_numbers += x.bitstring

    # convert to probability for each genome
    H = bitstr_numbers / len(pop)

    # calculate entropy, do not add 0 values
    H = -np.sum(np.where(H == 0, 0, H * np.log2(H)))

    return H


def plot_entropy(name: str, num_ep: int, entropies: list[float]):
    plt.clf()
    plt.plot(
        np.arange(num_ep),
        entropies,
        c="r",
        label="entropy",
    )
    plt.legend()
    plt.savefig(name)


def run(
    task: str,
    pop_size: int,
    num_ep: int,
    bitstr_size: int,
    plot_ep: int,
    num_parents: int,
    num_children: int,
    deterministic: bool,
    ranked: bool,
    crowding: bool,
):
    # create population
    population = create_population(size=pop_size, bitstr_size=bitstr_size, task=task)

    average_fitnesses = []
    entropies = []

    for episode in range(num_ep):
        average_fitness = np.mean(np.array([x.fitness for x in population]))
        # display average fitness
        print(episode, "average", average_fitness)
        # append to list for plotting
        average_fitnesses.append(average_fitness)
        # select best parents
        parents = parent_selection(
            population=population, num_parents=num_parents, task=task
        )

        if crowding:
            children = crossover(
                parents,
                bitstr_size=bitstr_size,
                num_children=num_children,
                deterministic=True,
                ranked=False,
                task=task,
            )

            survivors = crowding_tournament(parents, children, task)
        else:
            # perform crossover to get children
            children = crossover(
                parents,
                bitstr_size=bitstr_size,
                num_children=num_children,
                deterministic=deterministic,
                ranked=ranked,
                task=task,
            )

            # get survivors
            survivors = tournament(
                population=np.append(population, children).tolist(),
                pop_size=pop_size,
                task=task,
            )

        # update population
        population = survivors

        entropies.append(entropy(pop=population, bitstr_size=bitstr_size))

        if (episode + 1) % plot_ep == 0:
            plot(
                task=task,
                name=f"plots/{task}-{crowding}.png"
                if task == "g"
                else f"plots/{episode}-{task}-{crowding}.png",
                num_ep=episode + 1,
                avg_fit=average_fitnesses,
                pop=population,
            )
            plot_entropy(
                name=f"plots/entropy-{task}-{crowding}.png",
                num_ep=episode + 1,
                entropies=entropies,
            )

    plot(
        task=task,
        name=f"plots/{task}-{crowding}.png",
        num_ep=num_ep,
        avg_fit=average_fitnesses,
        pop=population,
    )
    plot_entropy(
        name=f"plots/entropy-{task}-{crowding}.png",
        num_ep=num_ep,
        entropies=entropies,
    )


if __name__ == "__main__":
    # CONFIG
    POPULATION_SIZE = 100
    NUM_EPISODES = 100
    BITSTRING_SIZE = 15

    PLOT_EP = 20

    # MUTATE_MIN = 0
    # MUTATE_MAX = 7
    # GENOME_MUTATE_CHANCE = 0.4

    TASK = "g"  # a,f,g
    if TASK == "g":
        BITSTRING_SIZE = 101

    run(
        task=TASK,
        pop_size=POPULATION_SIZE,
        num_ep=NUM_EPISODES,
        bitstr_size=BITSTRING_SIZE,
        plot_ep=PLOT_EP,
        num_parents=50,
        num_children=50,
        deterministic=False,
        ranked=True,
        crowding=True,
    )
