import math
import random
import numpy as np


class GA:
    def __init__(self, beta: float, threshold: float,
                 num_steps: int, num_hidden: int):

        # Initialising default values
        self.beta = beta
        self.beta_min = 0.05
        self.beta_max = 2.0
        self.beta_sigma = 'PH'

        self.threshold = threshold
        self.threshold_mix = 0.05
        self.threshold_max = 3.0
        self.threshold_sigma = 'PH'

        self.num_steps = num_steps
        self.num_steps_min = 5
        self.num_steps_max = 150
        self.num_steps_sigma = 'PH'

        self.num_hidden = num_hidden
        self.num_hidden_min = 50
        self.num_hidden_max = 2000
        self.num_hidden_sigma = 'PH'

    def random_float_gene(self, gene_current, gene_min, gene_max,
                          gene_sigma, mutation_type="FLOAT"): # Called by mutate
        if mutation_type == "FLOAT":
            mutated_value = random.gauss(gene_current, gene_sigma)
            #return np.random.uniform(gene_min, gene_max)
        elif mutation_type == "INT":
            mutated_value = math.floor(random.gauss(gene_current, gene_sigma))
        if mutated_value < gene_min:
            mutated_value = gene_min
        elif mutated_value > gene_max:
            mutated_value = gene_max

        return mutated_value

    def mutate(self) -> dict:
        self.beta =
        return {}

    def fitness(self, acc, watts) -> float:
        return 0.0

