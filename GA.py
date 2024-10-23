import time
import math

from random import gauss, randint

class GA:
    def __init__(self):

        # Initialising default values
        self.fitness_values = []
        self.fitness_rejection_threshold = 0.20
        self.prev_value_store = {"beta": [], "threshold": [], "num_steps": [], "num_hidden": []}

        self.sigma_multiplier = 0.20
        self.sigma_decay = 0.99
        self.sigma_boost = 1.01

        self.beta_mutation_type = "FLOAT"
        self.beta = 0.95
        self.beta_min = 0.05
        self.beta_max = 2.0
        self.beta_sigma = self.sigma_multiplier * self.beta_max

        self.threshold_mutation_type = "FLOAT"
        self.threshold = 1.0
        self.threshold_min = 0.05
        self.threshold_max = 3.0
        self.threshold_sigma = self.sigma_multiplier * self.threshold_max

        self.num_steps_mutation_type = "INT"
        self.num_steps = 25
        self.num_steps_min = 5
        self.num_steps_max = 100
        self.num_steps_sigma = self.sigma_multiplier * self.num_steps_max

        self.num_hidden_mutation_type = "INT"
        self.num_hidden = 1000
        self.num_hidden_min = 50
        self.num_hidden_max = 2000
        self.num_hidden_sigma = self.sigma_multiplier * self.num_hidden_max

    def _random_float_gene(self, gene): # Called by mutate
        if eval(f'self.{gene}_mutation_type') == "FLOAT":
            mutated_value = gauss(eval(f'self.{gene}'), eval(f'self.{gene}_sigma'))
        elif eval(f'self.{gene}_mutation_type') == "INT":
            mutated_value = math.floor(gauss(eval(f'self.{gene}'), eval(f'self.{gene}_sigma')))
        if mutated_value < eval(f'self.{gene}_min'):
            mutated_value = eval(f'self.{gene}_min')
        elif mutated_value > eval(f'self.{gene}_max'):
            mutated_value = eval(f'self.{gene}_max')
        return mutated_value

    def _mutate(self) -> dict:
        mutated_beta = self._random_float_gene("beta")
        mutated_threshold = self._random_float_gene("threshold")
        mutated_num_steps = self._random_float_gene("num_steps")
        mutated_num_hidden = self._random_float_gene("num_hidden")
        self.beta = mutated_beta
        self.threshold = mutated_threshold
        self.num_steps = mutated_num_steps
        self.num_hidden = mutated_num_hidden
        return self.beta

    def _adjust_sigma(self, adjustment):
        if adjustment == "decay":
            self.beta_sigma *= self.sigma_decay
            self.threshold_sigma *= self.sigma_decay
            self.num_steps_sigma *= self.sigma_decay
            self.num_hidden_sigma *= self.sigma_decay
        elif adjustment == "boost":
            self.beta_sigma *= self.sigma_boost
            self.threshold_sigma *= self.sigma_boost
            self.num_steps_sigma *= self.sigma_boost
            self.num_hidden_sigma *= self.sigma_boost

    def fitness(self, loss, watts, alpha=0.5) -> float:
        normalized_loss = 1 / (1 + loss)  # Higher fitness for lower loss
        normalized_watts = 1 / (1 + watts)  # Higher fitness for lower watts

        # Combine using weighted average
        fitness_score = alpha * normalized_loss + (1 - alpha) * normalized_watts
        # Minimum of 10 iterations to stabilise the SNN before we modify hyperparams
        if len(self.fitness_values) >= 10:
            if fitness_score - fitness_score * self.fitness_rejection_threshold <= self.fitness_values[-1]:
                # Use old values from an iteration prior and mutate again? Do I reduce sigma or check which values changed that caused a reduction
                # do that and use old values, calc difference between each and loss,
                # use this to determine how much to adjust sigma
                pass
            else:
                if fitness_score > self.fitness_values[-1]:
                    self._adjust_sigma("decay")
                else:
                    self._adjust_sigma("boost")
            self._mutate()
            self.fitness_values.append(fitness_score)
        else:
            self.fitness_values.append(fitness_score)
        return fitness_score

start = time.time()
i = 0
ga = GA()
while i < 50000:
    loss = randint(1, 30)/23
    watts = randint(1, 10)/10
    ga.fitness(loss, watts)
    print(ga.beta, ga.threshold, ga.num_steps, ga.num_hidden)
    i += 1