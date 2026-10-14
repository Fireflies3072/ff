from abc import ABC, abstractmethod
from typing import Any, Self
from collections.abc import Hashable
import copy
import numpy as np
from tqdm import tqdm

class EvolutionIndividual(ABC):
    def __init__(self, dna: Any):
        """Abstract base class for evolution individuals.

        Child classes must implement the mutate, crossover, and key methods.

        Args:
            dna: The DNA of the individual.
        """

        self.dna = dna
        self.fitness: float = -float('inf')
        self.stay: int = 0
        self.parent_key: Hashable|None = None

    @abstractmethod
    def mutate(self) -> Self:
        """Mutation operator: return the new mutated individual (deep copy isolation)."""
        pass

    @abstractmethod
    def crossover(self, other: Self) -> Self:
        """Crossover operator: crossover with another individual, return the child."""
        pass

    @abstractmethod
    def key(self) -> Hashable:
        """Return the unique hash key of the individual."""
        pass

    def clone(self) -> Self:
        """Deep copy of the individual."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} fitness={self.fitness:.4f}>"

class EvolutionStage(ABC):
    def __init__(self, keep_k: int):
        """Abstract base class for evolution stages.

        Child classes must implement the filter method.
        
        Args:
            keep_k: The number of individuals to keep.
        """
        
        self.keep_k = keep_k
        self.cache: dict[Hashable, Any] = {}

    @abstractmethod
    def filter(self, candidates: list[EvolutionIndividual], known_elites: dict[Hashable, EvolutionIndividual],
               pool: Any|None = None) -> list[EvolutionIndividual]:
        """Evaluate and filter the candidates based on the current stage's criteria.

        Args:
            candidates: The list of candidates to filter.
            known_elites: The dictionary of known elite individuals.
            pool: The multiprocessing pool to accelerate the evolution.

        Returns:
            The list of filtered individuals.
        """

        pass

class EvolutionPipeline:
    def __init__(self, stages: list[EvolutionStage], population_size: int = 8, candidate_size: int = 128):
        """ Connect the evolution stages and schedule the evolution.

        Args:
            stages: The list of evolution stages.
            population_size: The size of the population.
            candidate_size: The size of the candidate pool.
        """

        self.stages = stages
        self.population_size = population_size
        self.candidate_size = candidate_size

    def expand_population(self, population: list[EvolutionIndividual]) -> list[EvolutionIndividual]:
        """Expand the population by crossover and mutation to the candidate_size."""
        n = len(population)
        candidates = []
        seen_keys = {p.key() for p in population}

        def try_add(ind: EvolutionIndividual, parent_key: Hashable|None):
            k = ind.key()
            if k not in seen_keys:
                ind.parent_key = parent_key
                candidates.append(ind)
                seen_keys.add(k)

        # Crossover between all individuals in the population
        max_crossover_size = n * (n - 1) // 2
        expected_size = int(self.candidate_size * 0.3)
        if max_crossover_size <= expected_size:
            for i in range(n):
                for j in range(i + 1, n):
                    child = population[i].crossover(population[j])
                    # Record the individual with the higher fitness as parent_key
                    best_parent = population[i] if population[i].fitness >= population[j].fitness else population[j]
                    try_add(child, best_parent.key())
        elif n >= 2:
            # Calculate selection probabilities based on linear rank normalization
            sorted_population = sorted(population, key=lambda ind: ind.fitness)
            ranks = np.arange(1, n + 1, dtype=np.float64)
            probs = ranks / ranks.sum()

            attempts = 0
            while len(candidates) < expected_size and attempts < expected_size * 5:
                attempts += 1
                # Select 2 different parents without replacement
                idx1, idx2 = np.random.choice(n, size=2, replace=False, p=probs)
                ind1, ind2 = sorted_population[idx1], sorted_population[idx2]
                # Crossover the two individuals
                child = ind1.crossover(ind2)
                best_parent = ind1 if ind1.fitness >= ind2.fitness else ind2
                try_add(child, best_parent.key())
        crossover_size = len(candidates)
        
        # Mutate crossover children
        expected_size = int(self.candidate_size * 0.3)
        if crossover_size > 0:
            idx = 0
            max_attempts = expected_size * 10
            while len(candidates) < crossover_size + expected_size and idx < max_attempts:
                c = candidates[idx%crossover_size]
                child = c.mutate()
                try_add(child, c.key())
                idx += 1

        # Mutate parents
        idx = 0
        max_attempts = self.candidate_size * 10
        while len(candidates) < self.candidate_size and idx < max_attempts:
            p = population[idx%n]
            child = p.mutate()
            try_add(child, p.key())
            idx += 1
        
        return candidates

    def merge_with_elitism(self, previous_elites: list[EvolutionIndividual],
                           incoming_elites: list[EvolutionIndividual]) -> list[EvolutionIndividual]:
        """Classic (μ + λ) elitism strategy, update the stay count."""
        prev_keys = {ind.key() for ind in previous_elites}
        combined = {ind.key(): ind for ind in (previous_elites + incoming_elites)}

        # Rank the individuals by fitness and select the top self.population_size individuals
        ranked = sorted(combined.values(), key=lambda ind: ind.fitness, reverse=True)
        selected = ranked[:self.population_size]

        # Update the stay count
        for ind in selected:
            if ind.key() in prev_keys:
                ind.stay += 1
            else:
                ind.stay = 0
        return selected

    def step(self, population: list[EvolutionIndividual], is_initial: bool, pool: Any|None = None) -> list[EvolutionIndividual]:
        """Single evolution step: expand -> cascade stages -> merge with elites.
        
        Args:
            population: The population to start with.
            pool: The multiprocessing pool to accelerate the evolution.

        Returns:
            The new population after the evolution step.
        """

        known_elites = {ind.key(): ind for ind in population}

        # Expand the candidate pool
        if is_initial:
            candidates = population
        else:
            candidates = self.expand_population(population)

        # Pass through the stages
        current_pool = candidates
        for stage in self.stages:
            current_pool = stage.filter(current_pool, known_elites, pool=pool)

        # Merge with the historical elite individuals
        if is_initial:
            new_population = current_pool
        else:
            new_population = self.merge_with_elitism(population, current_pool)
        return new_population
    
    def evolve(self, population: list[EvolutionIndividual], num_generations: int, pool: Any|None = None,
               show_progress: bool = True) -> list[EvolutionIndividual]:
        """Evolve the population through the stages for a given number of generations.

        Generation 0 is only for evaluation. The following generations are for evolution.
        So num_generations + 1 is the total number of generations.

        Args:
            population: The initial population to start with.
            num_generations: The number of generations to evolve for.
            pool: The multiprocessing pool to accelerate the evolution.
            show_progress: Whether to show the progress bar.

        Returns:
            The list of evolved individuals.
        """

        for i in tqdm(range(num_generations + 1), disable=not show_progress):
            population = self.step(population, is_initial=(i==0), pool=pool)
        return population
