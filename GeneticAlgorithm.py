import numpy as np
import matplotlib.pyplot as plt

from LinReg import LinReg

# CONFIG
POPULATION_SIZE = 100
NUM_EPISODES = 1000
BITSTRING_SIZE = 7

MUTATE_MIN = 0
MUTATE_MAX = 7
GENOME_MUTATE_CHANCE = 0.2

TASK = "g"
if TASK == "g":
    BITSTRING_SIZE = 101

# solution/fitness
sine = np.sin(np.arange(0, 128))
linreg = LinReg()
data = np.loadtxt("14317816.txt", delimiter=",", encoding="UTF-8", dtype=float)


class Candidate:
    def __init__(self) -> None:
        # inverse binary repr
        self.bitstring: np.ndarray[int] = np.random.randint(0, 2, BITSTRING_SIZE)
        self.evaluate()

    def get_solution(self) -> int:
        """Gives the integer representation of the bitstring

        Returns:
            int: Integer to pass to sine function
        """
        num = 0
        for idx, bit_num in enumerate(self.bitstring):
            num += 2**idx * bit_num
        return int(num)

    def evaluate(self) -> float:
        # set fitness attribute
        if TASK == "a":
            fitness = sine[self.get_solution()]
        # f)
        elif TASK == "f":
            if not 4 < self.get_solution() < 11:
                fitness = sine[self.get_solution()]
                fitness -= 1

        # g)
        elif TASK == "g":
            columns = linreg.get_columns(X=data, bitstring=self.bitstring)
            fitness = linreg.get_fitness(x=columns, y=data[:, -1])
        self.fitness = fitness

    def mutate(
        self,
        genome_mutate_chance: float = GENOME_MUTATE_CHANCE,
        min_mutate_num: int = MUTATE_MIN,
        max_mutate_num: int = MUTATE_MAX + 1,
    ) -> None:
        # number of genomes to pick mutate
        num_genomes_to_mutate = np.random.randint(min_mutate_num, max_mutate_num)
        # select random places in bitstring
        genome_indices = np.random.choice(
            np.arange(len(self.bitstring)), num_genomes_to_mutate, replace=False
        )
        # invert bitstring in mutated spots
        for idx in genome_indices:
            if np.random.uniform() < genome_mutate_chance:
                self.bitstring[idx] = 1 - self.bitstring[idx]


def create_population(size) -> list[Candidate]:
    return [Candidate() for _ in range(size)]


def parent_selection(population: list[Candidate]) -> None:
    # sort population
    if TASK == "g":
        # get the worst
        population.sort(key=lambda x: x.fitness)
    else:
        # get the best
        population.sort(key=lambda x: x.fitness, reverse=True)
    # return the two first
    return population[:2]


def crossover(pair: np.ndarray[Candidate]) -> np.ndarray[Candidate]:
    # find point to split where to get dna from parents
    split_index = np.random.randint(1, BITSTRING_SIZE - 1)
    # create child1 and insert dna
    child1 = Candidate()
    child1.bitstring = np.append(
        pair[0].bitstring[:split_index], pair[1].bitstring[split_index:]
    )
    # create child2 and insert dna
    child2 = Candidate()
    child2.bitstring = np.append(
        pair[1].bitstring[:split_index], pair[0].bitstring[split_index:]
    )
    # mutate children
    if np.random.uniform() > 0.5:
        child1.mutate()
    if np.random.uniform() > 0.5:
        child2.mutate()

    child1.evaluate()
    child2.evaluate()

    return np.array([child1, child2])


def tournament(population: list[Candidate]) -> list[Candidate]:
    # sort population
    if TASK == "g":
        # the worst survive
        population.sort(key=lambda x: x.fitness, reverse=True)
    else:
        # the best survive
        population.sort(key=lambda x: x.fitness)

    # return survivors
    return population[2:]


def run():
    # create population
    population = create_population(POPULATION_SIZE)

    average_fitnesses = []

    for episode in range(NUM_EPISODES):
        average_fitness = np.mean(np.array([x.fitness for x in population]))
        # display average fitness
        print(episode, "average", average_fitness)
        # append to list for plotting
        average_fitnesses.append(average_fitness)

        # select best pair
        parents = parent_selection(population=population)

        # perform crossover to get children
        children = crossover(parents)

        # get survivors
        survivors = tournament(population=population)

        # update population
        population = np.append(survivors, children).tolist()

    if TASK == "g":
        plt.plot(
            np.arange(NUM_EPISODES), average_fitnesses, label="RMSE feature selected"
        )
        plt.plot(
            np.arange(NUM_EPISODES),
            np.ones(NUM_EPISODES) * linreg.get_fitness(data[:, :101], data[:, -1]),
            c="r",
            label="base RMSE",
        )
        plt.plot(
            np.arange(NUM_EPISODES), np.ones(NUM_EPISODES) * 0.124, c="g", label="0.124"
        )
        plt.legend()
    else:
        plt.plot(np.arange(128), sine)
        for i in population:
            plt.scatter(i.get_solution(), i.fitness, c="r")
    plt.show()


if __name__ == "__main__":
    run()
    # print("hello")
