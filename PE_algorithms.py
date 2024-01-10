import pandas as pd
import numpy as np
import random
from collections import defaultdict
import time
import math
from tqdm import tqdm
from dataclasses import dataclass
from typing import List


def brute_force_max_sum(sample_embeddings, num_proteins=10, full=False):
    """ Use parallel processing to Calculate the brute force max sum of Euclidean distances between embeddings

    :param sample_embeddings: dictionary containing an embedding label and embedding values
    :param num_proteins: how many diverse embeddings to select for the final subset
    :param full: whether the full dataset or a sample of the dataset is being used, if full save most diverse to csv
    :return: The list of embedding labels of the most diverse embeddings
    """
    # Convert the dictionary to a list of tuples
    sample_tuples = list(sample_embeddings.items())

    # list to store the diversity scores for each protein
    diversity_scores = []

    # Compute the total number of iterations
    total_iterations = len(sample_tuples) * (len(sample_tuples) - 1)

    # Initialize the progress bar
    progress_bar = tqdm(total=total_iterations, desc="Calculating diversity", unit="iteration")

    # Iterate over each protein embedding and label
    for label, embedding in sample_tuples:
        # Initialize the diversity score for the current protein
        diversity_score = 0

        # Calculate the sum of distances between the current protein and all other proteins
        for other_label, other_embedding in sample_tuples:
            if label != other_label:
                distance = np.linalg.norm(embedding - other_embedding)  # euclidean distance
                diversity_score += distance

            # Update the progress bar
            progress_bar.update()

        # Append the diversity score and protein label to the list
        diversity_scores.append((diversity_score, label))

    # Close the progress bar
    progress_bar.close()

    # Sort the diversity scores in descending order
    diversity_scores.sort(reverse=True)

    # If 'full' is True, save the results to a CSV file
    if full:
        df = pd.DataFrame(diversity_scores, columns=['MaxSum Value', 'Protein Label'])
        df.to_csv('top_proteins.csv', index=False)

    # Extract the top 'num_proteins' most diverse proteins
    top_proteins = [protein_label for _, protein_label in diversity_scores[:num_proteins]]

    return top_proteins


class MaxSumTabuSearch:
    """Alternative version of a Tabu search algorithm

    """
    def __init__(self, embeddings, num_proteins=10, max_iterations=1000, local_iterations=50, tabu_tenure=10):
        """Initialise the class

        :param embeddings: dict, the dataset
        :param num_proteins: int, number of top proteins to return
        :param max_iterations: int, max number of global iterations
        :param local_iterations: int, max number of local iterations
        :param tabu_tenure: int, number of iterations that a solution remains in tabu list
        """
        self.embeddings = embeddings
        self.num_proteins = num_proteins
        self.max_iterations = max_iterations
        self.local_iterations = local_iterations
        self.best_subset = None
        self.best_fitness = 0.0
        self.tabu_tenure = tabu_tenure
        self.tabu_list = {}

    def run_tabu_search(self):
        """Run an alternative tabu search
        Globally selects a small subset (num_proteins) then calls local search.
        Always randomly selects from global pool of potential 'solutions'.
        Updates best global fitness with best returned from local search.
        Keeps track of previously selected best (local and global) in tabu list,
        which is updated after every global iteration.  Updating either reduces the
        tabu tenure, or removes a solution from the list if it's tenure has dropped
        to 0 thus allowing the solution to be selected again.
        :return: list of most diverse proteins embeddings
        """
        progress_bar = tqdm(total=self.max_iterations, desc="Running Tabu Search")

        # Get the start time
        start_time = time.time()
        for _ in range(self.max_iterations):
            # Generate an initial solution with a random selection of n proteins
            initial_subset = random.sample(list(self.embeddings.keys()), self.num_proteins)

            # Local Search: Multi-start random sample subset of i proteins over j iterations
            improved_subset, fitness = self.local_search(initial_subset)

            # Update the best solution if a better one is found
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_subset = improved_subset

            progress_bar.update(1)  # Update the progress bar
            self.update_tabu_list()

            # Get the end time
            end_time = time.time()
            # Calculate the total elapsed time
            total_time = end_time - start_time
            print("Total time:", total_time)

        progress_bar.close()  # Close the progress bar after finishing
        return self.rank_proteins(self.best_subset)

    def rank_proteins(self, subset):
        """ sort the most diverse proteins by their max sum Euclidean distance in descending order

        :param subset: the subset of most diverse proteins
        :return: list of protein embeddings in descending order
        """
        # TODO remove this line as is not needed, already done in local_search
        individual_diversities = self.calculate_distances(subset)
        ranked_proteins = sorted(individual_diversities.keys(), key=lambda x: individual_diversities[x], reverse=True)
        return ranked_proteins[:self.num_proteins]

    def calculate_distances(self, subset):
        """ calculate the pairwise Euclidean distance of each protein embedding in the
        subset to all other embeddings in the subset

        :param subset: the subset of protein embeddings
        :return: dictionary of protein labels and their Euclidean distances
        """
        individual_diversities = {}
        for i, protein in enumerate(subset):
            total_distance = 0
            emb1 = self.embeddings[protein]
            for j, other_protein in enumerate(subset):
                if i != j:
                    emb2 = self.embeddings[other_protein]
                    distance = np.linalg.norm(emb1 - emb2)
                    total_distance += distance
            individual_diversities[protein] = total_distance
        return individual_diversities

    def local_search(self, subset):
        """Search local populations for most diverse proteins
        Selects a list of 100 random proteins from the global search space and
        at each iteration evaluates the solution, keeping track of the best local
        solution.
        :param subset: current best solution
        :return: The best solution and their fitnessess
        """
        best_subset = subset
        best_fitness = self.max_sum_fitness(subset)

        for _ in range(self.local_iterations):
            # take a random sample of 100 proteins
            random_sampled_prot = random.sample(list(self.embeddings.keys()), 100)
            # calculate fitness
            fitness = self.max_sum_fitness(random_sampled_prot)

            solution_tuple = tuple(sorted(random_sampled_prot))

            # Even if a solution is tabu, it can be accepted if it's better than the best known solution.
            if (solution_tuple not in self.tabu_list) or (fitness > self.best_fitness):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_subset = random_sampled_prot
                    # Add to tabu list with tabu tenure
                    self.tabu_list[solution_tuple] = self.tabu_tenure

        return best_subset, best_fitness

    def max_sum_fitness(self, protein_labels):
        """Sum the distances and return the total distance

        :param protein_labels: dictionary where keys are protein labels and values are a list of distances
        :return: total distance
        """
        total_distance = sum(self.calculate_distances(protein_labels).values())
        return total_distance

    def update_tabu_list(self):
        """Decrement the tenure of each solution in the tabu list. Remove solutions with zero tenure."""
        to_remove = [sol for sol, tenure in self.tabu_list.items() if tenure <= 1]
        for sol in to_remove:
            del self.tabu_list[sol]

        for sol in self.tabu_list:
            self.tabu_list[sol] -= 1


class TradMaxSumTabuSearchV1:
    """Traditional tabu search implementation with simple yet conservative population generation """
    def __init__(self, embeddings, num_proteins=10, max_iterations=1000, tabu_tenure=10):
        """Initialise the class

        :param embeddings: dict, the dataset {protein:[embeddings], protein:[embeddings], ...}
        :param num_proteins: int, number of top proteins to return
        :param max_iterations: int, max number of global iterations
        :param tabu_tenure: int, number of iterations that a solution remains in tabu list
        """
        self.embeddings = embeddings
        self.num_proteins = num_proteins
        self.max_iterations = max_iterations
        self.best_subset = None
        self.best_fitness = 0.0
        self.tabu_list = set()
        self.tabu_tenure = tabu_tenure

    def run_tabu_search(self):
        """ Run an alternative tabu search
        Selects an initial subset (num_proteins) which gets passed to local search
        where a neighbourhood of solutions are created by swapping out a single protein
        label in the solution list with a random sample from the global list.
        :return: list of most diverse proteins embeddings
        """
        progress_bar = tqdm(total=self.max_iterations, desc="Running Tabu Search")

        # Get the start time
        start_time = time.time()
        for _ in range(self.max_iterations):
            # Generate an initial solution with a random selection of 10 proteins
            initial_subset = random.sample(list(self.embeddings.keys()), self.num_proteins)

            #  Multi-start random sample subset of n proteins over j iterations
            improved_subset, fitness = self.local_search(initial_subset)

            # Update the best solution if a better one is found
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_subset = improved_subset

            progress_bar.update(1)  # Update the progress bar

            # Get the end time
            end_time = time.time()
            # Calculate the total elapsed time
            total_time = end_time - start_time
            print("Total time:", total_time)

        progress_bar.close()  # Close the progress bar after finishing
        return self.rank_proteins(self.best_subset)

    def rank_proteins(self, subset):
        """Rank based on highest max sum

        :param subset: the subset of protein embeddings
        :return: the ranked proteins
        """
        ranked_proteins = sorted(subset, reverse=True)
        return ranked_proteins

    def local_search(self, subset):
        """ Create a neighborhood and conduct a local search to find the most diverse proteins
        Only uses a local tabu list.
        :param subset: global solution space
        :return: tuple (best subset, best fitness)
        """
        best_subset = subset
        best_fitness = self.max_sum_fitness(subset)
        tabu_list = {}  # dictionary to store tabu solutions and their remaining tabu tenure

        for _ in range(50):
            neighborhood = self.generate_neighborhood(best_subset)

            # Evaluate the fitness of solutions in the neighborhood
            neighborhood_fitness = [self.max_sum_fitness(solution) for solution in neighborhood]

            # Find the best non-tabu solution in the neighborhood
            for i, solution in enumerate(neighborhood):
                solution_tuple = tuple(solution)  # Convert the list to a tuple
                if solution_tuple not in tabu_list or tabu_list[solution_tuple] <= 0:
                    if neighborhood_fitness[i] > best_fitness:
                        best_fitness = neighborhood_fitness[i]
                        best_subset = solution
                    tabu_list[solution_tuple] = self.tabu_tenure  # Use the tuple as the key
                    # break - exits early and might not find the best as these are not sorted

            # Decrement tabu tenure for all solutions in the tabu list
            for solution, tenure in tabu_list.items():
                tabu_list[solution] = tenure - 1

        return best_subset, best_fitness

    def generate_neighborhood(self, subset):
        """Generate a set of new solutions by randomly changing an element from the initial random solution
        A new solution (list) in the subset neighborhood space is the same subset list with one
        item swapped out with a randomly selected label from the global solution space.
        :param subset: current or initial solution to generate new solutions from
        :return: list of new solutions / a neighbourhood
        """
        neighborhood = []
        for i in range(len(subset)):
            # Generate solutions by making a single change to the current solution
            new_solution = subset[:i] + [random.choice(list(self.embeddings.keys()))] + subset[i + 1:]
            neighborhood.append(new_solution)
        return neighborhood

    def max_sum_fitness(self, protein_embeddings):
        """Calculate the max sum pairwise Euclidean distance between each embedding and all other embeddings

        :param protein_embeddings: the protein embeddings
        :return: the max sum of the embeddings
        """
        total_distance = 0

        # Calculate the sum of distances between embeddings in the subset
        for i in range(len(protein_embeddings)):
            emb1 = self.embeddings[protein_embeddings[i]]
            for j in range(len(protein_embeddings)):
                emb2 = self.embeddings[protein_embeddings[j]]
                # Euclidean distance
                distance = np.linalg.norm(emb1 - emb2)
                total_distance += distance

        return total_distance


class TradMaxSumTabuSearchV2:
    """ Traditional tabu search to find the most diverse proteins embeddings """
    def __init__(self, embeddings, num_proteins=10, max_iterations=1000,
                 local_iterations=100, local_sample_sizes=100):
        """Initialise the class

        :param embeddings: dict, the dataset {protein:[embeddings], protein:[embeddings], ...}
        :param num_proteins: int, number of top proteins to return
        :param max_iterations: int, max number of global iterations
        :param local_iterations: int, max number of local iterations
        :param local_sample_sizes: the size of the local solution / num proteins in list
        """
        self.embeddings = embeddings
        self.num_proteins = num_proteins
        self.max_iterations = max_iterations
        self.best_subset = None
        self.best_fitness = 0.0
        self.tabu_list = set()
        self.lss = local_sample_sizes
        self.local_iterations = local_iterations
        self.global_sampling_number = int(len(embeddings) / 20)
        self.tabu_tenure = 10

    def run_tabu_search(self):
        """Run the tabu search on the protein embeddings and find the most diverse.

        Initialises on an initial subset (num_proteins) from the global solution space.
        Select a smaller subset (global embeddings / 20) which serves as a local
        solution space to explore.  The best solution from local search is checked
        for fitness against global solution set.
        :return: list of the protein labels of the most diverse proteins, global history and local history
        to see how often labels get picked
        """
        local_hist = defaultdict(int)  # for analysis of how many times labels are selected
        global_hist = defaultdict(int)
        progress_bar = tqdm(total=self.max_iterations, desc="Running Tabu Search")

        # Generate an initial solution
        initial_subset = random.sample(list(self.embeddings.keys()), self.num_proteins)
        # Calculate fitness
        fitness = self.max_sum_fitness(initial_subset)
        self.check_fitness(fitness, initial_subset, global_hist)

        # Get the start time
        start_time = time.time()
        for _ in range(self.max_iterations):
            # get a large sample from the global embeddings to subsample in local search from
            global_subset = random.sample(list(self.embeddings.keys()), self.global_sampling_number)
            # Multi-start random sample subset of 1000 proteins over 1000 iterations
            improved_subset, fitness, local_hist = self.local_search(global_subset, local_hist, self.lss)
            # Update the best solution if a better one is found
            self.check_fitness(fitness, improved_subset, global_hist)
            progress_bar.update(1)  # Update the progress bar

            # Get the end time
            end_time = time.time()
            # Calculate the total elapsed time
            total_time = end_time - start_time
            print("Total time:", total_time)

        progress_bar.close()  # Close the progress bar after finishing
        return self.rank_proteins(self.best_subset), global_hist, local_hist

    def rank_proteins(self, subset):
        """Calculate the sum of the pairwise Euclidean distance and rank the proteins based on the max sum

        :param subset: list of proteins labels unsorted
        :return: ranked list of proteins labels
        """
        ranked_proteins = sorted(subset, reverse=True)
        return ranked_proteins[:self.num_proteins]

    def check_fitness(self, fitness, subset, history):
        """check the fitness of the subset and update the best fitness
        and history tracker

        :param fitness: max sum euclidean distance
        :param subset: dictionary of proteins embeddings
        :param history: global history stored in a dictionary
        """
        if fitness > self.best_fitness:
            print("improved the subset")
            self.best_fitness = fitness
            self.best_subset = subset
            for item in self.best_subset:
                history[item] += 1

    def local_search(self, subset, histo, samp_size):
        """Explore the local search space for the best solution subset (list of most diverse proteins)

        Create a solution subset of local sample size, randomly sampled from the local subset and
        evaluate its fitness, do this for x iterations.  Only uses a local tabu list
        :param subset: the local subset of embeddings to explore, randomly selected from the global solution space
        :param histo: dict, local history of previously selected proteins - for analysis of random selection
        :param samp_size: the local solution list size
        :return: tuple(the best subset, the best fitness, selection history)
        """
        best_fitness = 0
        best_subset = None
        tabu_list = defaultdict(int)
        for _ in range(self.local_iterations):
            # Get a random sample of the local subset dataset
            random_sampled_prot = random.sample(subset, samp_size)
            # Calculate fitness for the current random sample subset
            fitness = self.max_sum_fitness(random_sampled_prot)

            # Update the best solution if a better one is found
            if fitness > best_fitness:
                if tuple(random_sampled_prot) not in tabu_list.keys() or tabu_list[tuple(random_sampled_prot)] == 0:
                    best_fitness = fitness
                    best_subset = random_sampled_prot
                    tabu_list[tuple(best_subset)] = self.tabu_tenure  # Set the tabu tenure for the chosen solution

            # Decrement tabu tenure for all solutions in the tabu list
            for solution in tabu_list:
                tabu_list[solution] -= 1

        return best_subset, best_fitness, histo

    def max_sum_fitness(self, protein_labels):
        """Sum the distances and return the total distance

        :param protein_labels: dictionary where keys are protein labels and values are a list of distances
        :return: total distance
        """
        total_distance = 0

        # Calculate the sum of distances between embeddings in the subset
        for i in range(len(protein_labels)):
            emb1 = self.embeddings[protein_labels[i]]
            for j in range(len(protein_labels)):
                emb2 = self.embeddings[protein_labels[j]]
                # Euclidean distance
                distance = np.linalg.norm(emb1 - emb2)
                total_distance += distance

        return total_distance


@dataclass
class ProteinSolution:
    """Encapsulate the solution list and its fitness in an object for
    easier genetic algorithm handling"""
    fitness: float
    proteins: List[str]  # list of protein names


class MemeticGLS:
    def __init__(self, embeddings, dna_size=20, max_epochs=1000, local_iterations=20, population_size=1000,
                 retain_percent=0.05):
        """Genetic Local Search to solve the Maximum Diversity Problem in relation to finding the list
        of x most diverse protein embeddings

        Globals search with genetic algorithm, local search with Tabu search
        :param embeddings: dict, {protein_name: embedding}
        :param dna_size: int, solution (most diverse protein list size) how many max diverse proteins to find
        :param max_epochs: int, maximum number of iterations to run
        :param local_iterations: int, maximum number of local refinement iterations
        :param population_size: int, number of individuals in a population
        """
        self.embeddings = embeddings
        self.dna_sze = dna_size
        self.max_epochs = max_epochs
        self.best_subset = None
        self.best_fitness = 0.0
        self.local_iters = local_iterations
        self.pop_size = population_size
        self.tabu_tenure = 10
        self.best_retain = math.ceil(retain_percent * self.pop_size)

    def evolve_solution(self):
        """Search for most diverse proteins in the global search space using a genetic algorithm approach

        creates new population by:
        1) selecting the best x num from existing population,
        2) crossing over the best x num from existing population with each other by taking the first half
            of parent 1's dna (solution list) and the second half of parent 2's dna to make a new individual,
        3) mutating the next best individuals (lower ranked individuals and twice the x best to retain),
            mutate from the global solution space,
        4) refine the next x individuals lower down the rankings. Refining mutates the individuals with
            randomly selected proteins from the local search space and uses a tabu list to ensure that
            no duplicates are added to the new population
        6) the rest of the new population is randomly selected from the global space
        :return: The best solution, list of all best individuals
        """
        best_individuals = []
        for epoch in range(self.max_epochs):
            print(f"** EPOCH {epoch} out of {self.max_epochs}")
            # update the size of the new population by subtracting the number
            # of the best individuals (selected in the previous epoch) from the new population size
            pop_size = self.pop_size - len(best_individuals)
            population, local_space = self.generate_population(pop_size)
            # now add the best individuals to population
            population.extend(best_individuals)
            # calculate the MaxSum of each individual in the population
            self.calc_fitness(population)
            # Rank individuals in the population based on their fitness
            ranked_individuals = self.rank_individuals(population)
            # terminate once epochs have come to an end - no need to do anymore evolving
            if (epoch + 1) == self.max_epochs:
                best_proteins = ranked_individuals[0].proteins
                return best_proteins, ranked_individuals

            # new population creation
            # select best x individuals from this population for next epoch
            new_population = ranked_individuals[:self.best_retain]
            # select x individuals for crossover - crossover from global embedding space
            new_population.extend(self.crossover(ranked_individuals[:self.best_retain]))
            # select x individuals for mutation - mutate from global embedding space
            mutate_start = self.best_retain
            mutate_end = self.best_retain * 2
            new_population.extend(self.mutate(ranked_individuals[mutate_start:mutate_end],
                                              list(self.embeddings.keys())))
            # do some refining from local space
            # refine x individuals with local search - select individuals a bit further down the list
            mutate_start = mutate_end
            mutate_end = mutate_end + self.best_retain * 2
            new_population.extend(self.tabu_refine(ranked_individuals[mutate_start:mutate_end],
                                                   local_space, ranked_individuals[:self.best_retain]))
            best_individuals = new_population

    def generate_population(self, population_size):
        """Randomly create a population of solutions, these are a sublist of protein embeddings selected from the
        global embedding space
        :param population_size: int the population size to generate
        :return list[ProteinSolutions]
        """
        population = []
        #   a set to store the unique keys of the current solution space
        local_space = set()

        for i in range(population_size):
            embeddings_list = random.sample(list(self.embeddings.keys()), self.dna_sze)
            population.append(ProteinSolution(fitness=0.0, proteins=embeddings_list))
            local_space.update(embeddings_list)

        return population, local_space

    def calc_fitness(self, population):
        """Calculate the max sum of the distances of all the embeddings in each individual solution
        from each other and set the fitness on the individual

        :param population: the list of protein solutions in the population for a single epoch
        :return: None
        """
        # print("In calf fitness and population is ", population)
        for individual in population:
            # list of protein labels
            solution_subset = individual.proteins
            for i in range(len(solution_subset)):
                # get the actual embeddings array using the protein label
                chromosome1 = self.embeddings[solution_subset[i]]
                for j in range(len(solution_subset)):
                    chromosome2 = self.embeddings[solution_subset[j]]
                    # Euclidean distance
                    distance = np.linalg.norm(chromosome1 - chromosome2)
                    individual.fitness += distance

    def rank_individuals(self, population):
        """Sort the population by fitness
        :param population: current population
        :return: the sorted population
        """
        return sorted(population, key=lambda x: x.fitness, reverse=True)

    def crossover(self, individuals):
        """Create a new set of solutions by performing cross over on individuals in the list
        Should end up with only 1 child per two parents - can also do the reverse and end up with two children
        take first half of parent 1's dna and add second half of parent 2's dna, can do vice versa
        :param individuals: set of individuals to perform cross over on
        :return: the list of new child individuals
        """
        # assert that the list is even
        new_individuals = []
        for i in range(len(individuals), 2):
            protein_parent_1 = individuals[i]
            if i + 1 < len(individuals):
                protein_parent_2 = individuals[i + 1]
            else:
                # end of the uneven list, use the first protein solution for crossover again
                protein_parent_2 = individuals[0]

            proteins_cut_off = self.dna_sze / 2
            new_proteins_list = protein_parent_1.proteins[:proteins_cut_off] + protein_parent_2[proteins_cut_off:]
            new_individuals.append(ProteinSolution(fitness=0, proteins=new_proteins_list))

        return new_individuals

    def mutate(self, individuals, emb):
        """Create a new set of solutions by mutation
        Only ever mutate to a maximum of 75% of the current individual's dna
        Randomly mutate each individual's dna - so the rate that each individual
        is mutated by is different and randomly chosen between 1 and max limit
        :param individuals: the individuals to mutate
        :param emb: the embedding space to randomly sample from
        :return:
        """
        # mutate no more that 75% of the current solution
        # math.ciel rounds it up to the next nearest int
        mutation_limit = math.ceil(self.dna_sze * 0.75)

        for individual in individuals:
            # select a random number of chromosomes to mutate
            proteins_to_mutate = random.randint(1, mutation_limit)
            # Select where along the chromosomes to snip/ not mutate
            protein_cut_off = self.dna_sze - proteins_to_mutate

            new_proteins = random.sample(emb, proteins_to_mutate)
            # mutate
            individual.proteins = individual.proteins[:protein_cut_off] + new_proteins
            # get ready for new fitness check
            individual.fitness = 0

        return individuals

    def tabu_refine(self, individuals, local_space, tabu_list):
        """Refine the list of individuals to create a new set of solutions based on mutation
        with x number of proteins from the local space, ensure that the new solutions are not in the tabu list
        the tabu list is the list of best individuals

        :param individuals: the list of solutions to refine
        :param local_space: the randomly selected subset for this epoch
        :param tabu_list: the list of the best individuals already in the new population
        :return refined_individuals: list of refined, through local mutation, individuals
        """
        tabu_list = [x.proteins for x in tabu_list]
        recreate_list = individuals
        return_size = len(recreate_list)
        max_retries = self.local_iters
        new_population = []
        while max_retries > 0:
            refined_individuals = self.mutate(recreate_list, local_space)
            recreate_list = []
            for individual in refined_individuals:
                if individual.proteins in tabu_list:
                    recreate_list.append(individual)
                else:
                    new_population.append(individual)
            if len(new_population) == return_size:
                return new_population
            max_retries -= 1
        return new_population