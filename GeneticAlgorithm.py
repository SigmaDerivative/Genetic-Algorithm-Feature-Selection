import copy

import numpy as np
import matplotlib.pyplot as plt


class Candidate:
    def __init__(self) -> None:
        # inverse binary repr
        self.bitstring = np.random.randint(0, 2, 7)
        self.score = -1

    def get_solution(self) -> int:
        """Gives the integer representation of the bitstring

        Returns:
            int: Integer to pass to sine function
        """
        num = 0
        for idx, a in enumerate(self.bitstring):
            num += 2**idx * a
        return int(num)

    def combine(self, partner) -> None:
        # number of genomes to pick from partner
        num_genomes_to_pick = np.random.randint(1, 7)
        # select random places in bitstring
        genome_indices = np.random.choice(
            np.arange(len(self.bitstring)), num_genomes_to_pick, replace=False
        )
        # take the partner bitstr values
        self.bitstring[genome_indices] = partner.bitstring[genome_indices]

    def mutate(self) -> None:
        # number of genomes to pick mutate
        num_genomes_to_mutate = np.random.randint(0, 3)
        # select random places in bitstring
        genome_indices = np.random.choice(
            np.arange(len(self.bitstring)), num_genomes_to_mutate, replace=False
        )
        # invert bitstring in mutated spots
        self.bitstring[genome_indices] = 1 - self.bitstring[genome_indices]


def run():
    pop_size = 100
    num_episodes = 1000

    # solution/fitness
    sine = np.sin(np.arange(0, 128))

    # list of candidates
    candidates = []
    for _ in range(pop_size):
        candidates.append(Candidate())

    average_scores = []

    for episode in range(num_episodes):
        scores = []
        for idx, can in enumerate(candidates):
            # get score
            candidate_score = sine[can.get_solution()]
            can.score = candidate_score
            scores.append(candidate_score)
            # print(idx + 1, can.get_solution(), can.score)
        print(episode, "average", np.mean(scores))
        average_scores.append(np.mean(scores))

        # randomly select a pair
        pair_idx = np.random.choice(np.arange(0, pop_size), 2, replace=False)
        baby = copy.deepcopy(candidates[pair_idx[0]])
        baby.combine(candidates[pair_idx[1]])
        baby.mutate()

        # get idx of worst candidate
        worst_can_idx = np.argmin(scores)

        # inject baby into population position of worst candidate
        candidates[worst_can_idx] = baby

    # print(num_episodes)
    plt.plot(np.arange(num_episodes), average_scores)
    plt.show()


if __name__ == "__main__":
    run()
