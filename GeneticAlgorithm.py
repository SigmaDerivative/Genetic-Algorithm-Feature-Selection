import numpy as np
import matplotlib.pyplot as plt

# CONFIG
POPULATION_SIZE = 100
NUM_EPISODES = 1000
BITSTRING_SIZE = 7

MUTATE_MIN = 0
MUTATE_MAX = 7
GENOME_MUTATE_CHANCE = 0.2

# solution/fitness
sine = np.sin(np.arange(0, 128))


class Candidate:
    def __init__(self) -> None:
        # inverse binary repr
        self.bitstring: np.ndarray[int] = np.random.randint(0, 2, BITSTRING_SIZE)
        self.fitness: int = -1

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
        fitness = sine[self.get_solution()]
        self.fitness = fitness
        return fitness

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


def evaluate_population(population: list[Candidate]) -> float:
    # evaluate population
    fitnesses = []
    for candidate in population:
        fitnesses.append(candidate.evaluate())
    # return mean
    return np.mean(fitnesses)


def parent_selection(population: list[Candidate], best: bool = True) -> None:
    # sort population
    population.sort(key=lambda x: x.fitness, reverse=True)
    if best:
        return population[:2]
    # random
    return np.random.choice(population, 2, replace=False)


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

    # print(child1.evaluate())

    return np.array([child1, child2])


def tournament(population: list[Candidate]) -> list[Candidate]:
    # sort population
    population.sort(key=lambda x: x.fitness)

    # return survivors
    return population[2:]


def run():

    # create population
    population = create_population(POPULATION_SIZE)

    average_fitnesses = []

    for episode in range(NUM_EPISODES):
        # rank population and get average fitness
        average_fitness = evaluate_population(population=population)

        # display average fitness
        print(episode, "average", average_fitness)
        # append to list for plotting
        average_fitnesses.append(average_fitness)

        # select best pair
        parents = parent_selection(population=population, best=True)

        # perform crossover to get children
        children = crossover(parents)

        # get survivors
        survivors = tournament(population=population)

        # update population
        population = np.append(survivors, children).tolist()

    # print(num_episodes)
    plt.plot(np.arange(NUM_EPISODES), average_fitnesses)
    plt.show()


if __name__ == "__main__":
    run()
