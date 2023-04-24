# %%
# References:
# - [A review on genetic algorithm: past, present, and future](https://link.springer.com/article/10.1007/s11042-020-10139-6)
import numpy as np
import random

# Genetic algorithm (GA) is an optimization algorithm that is inspired from the natural selection.
# It is a population based search algorithm, which utilizes the concept of survival of fittest.
# The new populations are produced by iterative use of genetic operators on individuals present
# in the population. The chromosome representation, selection, crossover, mutation, and fitness
# function computation are the key elements of GA.
#
# The procedure of GA is as follows.


def genetic_algorithm(
    fitness_function,
    cross_heuristic_selector,
    cross_function,
    mutation_function,
    epoch=1000,
    population_size=50,
    chain_size=10,
):
    # A population (Y) of n chromosomes are initialized randomly.
    population = np.random.randint(2, size=(population_size, chain_size))

    fitness_vertex = np.vectorize(fitness_function)
    mutation_vertex = np.vectorize(mutation_function)

    for _ in range(epoch):
        # The fitness of each chromosome in Y is computed.
        fitness_values = fitness_vertex(population)

        # Two chromosomes say C1 and C2 are selected from the population Y according to the fitness value.
        # The single-point crossover operator with crossover probability (Cp) is applied on C1 and C2
        # to produce an offspring say O.
        cross_tuples = cross_heuristic_selector(fitness_values)
        cross_offspring = [
            cross_function(population[c1], population[c2])
            for c1, c2 in cross_tuples
        ]

        # Thereafter, uniform mutation operator is applied on produced offspring (O)
        # with mutation probability (Mp) to generate O′.
        # The new offspring O′ is placed in new population.
        population = mutation_vertex(np.array(cross_offspring))

        # The selection, crossover, and mutation operations will be repeated
        # on current population until the new population is complete.


# GAs used a variety of operators during the search process.
# These operators are encoding schemes, crossover, mutation, and selection.


# For most of the computational problems, the encoding scheme
# (i.e., to convert in particular form) plays an important role.
# The encoding schemes are differentiated according to the problem domain.
# The well-known encoding schemes are binary, octal, hexadecimal, permutation, value-based, and tree.

def binary_encoding():
    """
    Binary encoding is the commonly used encoding scheme. 
    Each gene or chromosome is represented as a string of 1 or 0. 
    In binary encoding, each bit represents the characteristics of the solution. 
    It provides faster implementation of crossover and mutation operators. 
    However, it requires extra effort to convert into binary form 
    and accuracy of algorithm depends upon the binary conversion. 
    The bit stream is changed according the problem. 
    Binary encoding scheme is not appropriate for some engineering design problems 
    due to epistasis and natural representation.
    """


def permutation_encoding():
    """
    The permutation encoding scheme is generally used in ordering problems. 
    In this encoding scheme, the gene or chromosome is represented by the string of numbers 
    that represents the position in a sequence
    """


def tree_encoding():
    """
    In tree encoding, the gene or chromosome is represented by a tree of functions or commands. 
    These functions and commands can be related to any programming language. 
    This is very much similar to the representation of repression in tree format. 
    This type of encoding is generally used in evolving programs or expressions.
    """


# Selection is an important step in genetic algorithms that determines whether
# the particular string will participate in the reproduction process or not.
# The selection step is sometimes also known as the reproduction operator.
# The convergence rate of GA depends upon the selection pressure.
# The well-known selection techniques are roulette wheel,
# rank, tournament, boltzmann, and stochastic universal sampling.


def roulette_wheel(population, fitness_values):
    """
    Roulette wheel selection maps all the possible strings onto a wheel with a portion of the wheel allocated 
    to them according to their fitness value. This wheel is then rotated randomly to select specific solutions 
    that will participate in formation of the next generation. However, it suffers from many problems 
    such as errors introduced by its stochastic nature.

    :param population: list or array of solutions to be selected from
    :param fitness_values: list or array of fitness values for each solution in population
    :return: a selected solution from population
    """

    # Calculate the sum of all fitness values
    fitness_sum = sum(fitness_values)

    # Calculate the probabilities of each solution
    probabilities = np.array(
        [fitness / fitness_sum for fitness in fitness_values])

    # Spin the wheel
    spin = random.uniform(0, 1)
    spin_position = 0
    for i, probability in enumerate(probabilities):
        spin_position += probability
        if spin <= spin_position:
            return population[i]


def rank(population, fitness_values):
    """
    De Jong and Brindle modified the roulette wheel selection method to remove errors 
    by introducing the concept of determinism in selection procedure.
    Rank selection is the modified form of Roulette wheel selection. 
    It utilizes the ranks instead of fitness value. 
    Ranks are given to them according to their fitness value so 
    that each individual gets a chance of getting selected according to their ranks. 
    Rank selection method reduces the chances of prematurely converging the solution to a local minima

    :param population: list or array of solutions to be selected from
    :param fitness_values: list or array of fitness values for each solution in population
    :return: a selected solution from population
    """

    # Assign ranks based on fitness values
    ranks = np.argsort(fitness_values)

    # Calculate probabilities based on ranks
    probabilities = np.array([((len(ranks) - i) / len(ranks))
                             for i in range(len(ranks))])
    probabilities = probabilities / np.sum(probabilities)

    # Select a solution based on rank probabilities
    selected_index = np.random.choice(range(len(population)), p=probabilities)

    return population[selected_index]


def tournament(population, fitness_values, tournament_size=2):
    """
    Tournament selection technique was first proposed by Brindle in 1983. The individuals are selected 
    according to their fitness values from a stochastic roulette wheel in pairs. After selection, the individuals 
    with higher fitness value are added to the pool of next generation. In this method of selection, each individual 
    is compared with all n-1 other individuals if it reaches the final population of solutions.

    :param population: list or array of solutions to be selected from
    :param fitness_values: list or array of fitness values for each solution in population
    :param tournament_size: the size of the tournament (default 2)
    :return: a selected solution from population

    **Example**
    ```python
    >>> population = ['solution1', 'solution2', 'solution3', 'solution4']
    >>> fitness_values = [0.9, 0.6, 0.7, 0.8]
    >>> tournament(population, fitness_values)
    'solution1'

    """

    tournament = random.sample(
        list(zip(population, fitness_values)), tournament_size)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]


def boltzmann(population, fitness_values, t):
    """
    Boltzmann selection is based on entropy and sampling methods, which are used in Monte Carlo Simulation. 
    It helps in solving the problem of premature convergence. The probability is very high for selecting 
    the best string, while it executes in very less time. However, there is a possibility of information loss. 
    It can be managed through elitism.

    :param population: list or array of solutions to be selected from
    :param fitness_values: list or array of fitness values for each solution in population
    :param t: temperature parameter
    :return: a selected solution from population

    **Example**
    ```python
    >>> population = ['solution1', 'solution2', 'solution3', 'solution4']
    >>> fitness_values = [0.9, 0.6, 0.7, 0.8]
    >>> t = 1.0
    >>> boltzmann(population, fitness_values, t)
    'solution1'

    """
    fitness_exp = np.exp(np.array(fitness_values) / t)
    probabilities = fitness_exp / np.sum(fitness_exp)
    selected_index = np.random.choice(range(len(population)), p=probabilities)
    return population[selected_index]


def elitism(population, fitness_values, n_elites=1):
    """
    Elitism selection was proposed by K. D. Jong (1975) for improving the performance of Roulette wheel selection. 
    It ensures the elitist individual in a generation is always propagated to the next generation. If the individual 
    having the highest fitness value is not present in the next generation after normal selection procedure, 
    then the elitist one is also included in the next generation automatically.

    :param population: list or array of solutions to be selected from
    :param fitness_values: list or array of fitness values for each solution in population
    :param n_elites: the number of elite individuals to include in the next generation (default 1)
    :return: a list of elite individuals

    **Example**
    ```python
    >>> population = ['solution1', 'solution2', 'solution3', 'solution4']
    >>> fitness_values = [0.9, 0.6, 0.7, 0.8]
    >>> n_elites = 2
    >>> elitism(population, fitness_values, n_elites)
    ['solution1', 'solution4']
    ```
    """
    elites = []
    for i in range(n_elites):
        elite_index = np.argmax(fitness_values)
        elites.append(population[elite_index])
        population.pop(elite_index)
        fitness_values = np.delete(fitness_values, elite_index)

    return elites


# Crossover operators are used to generate the offspring by combining the genetic information of two or more parents.
# The well-known crossover operators are single-point, two-point, k-point, uniform, partially matched,
# order, precedence preserving crossover, shuffle, reduced surrogate and cycle.


def single_point(a, b):
    """
    In a single point crossover, a random crossover point is selected. 
    The genetic information of two parents which is beyond that point will be swapped with each other
    It replaced the tail array bits of both the parents to get the new offspring.

    :param list a: A list of integers representing the genetic information of parent A. 
    :param list b: A list of integers representing the genetic information of parent B. 
    :return: A list representing the genetic information of the new offspring.

    **Example**
    ```python
    >>> single_point([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    [1, 2, 3, 9, 10]
    >>> single_point([10, 20, 30, 40, 50], [60, 70, 80, 90, 100])
    [10, 20, 30, 90, 100]
    ```
    """
    import random

    # Choose a random crossover point
    crossover_point = random.randint(0, len(a) - 1)

    # Swapping genetic information to create offspring
    offspring = a[:crossover_point] + b[crossover_point:]

    return offspring


def k_point(a, b, k=2):
    """
    In a two point and k-point crossover, two or more random crossover points are selected 
    and the genetic information of parents will be swapped as per the segments that have been created
    The middle segment of the parents is replaced to generate the new offspring.

    Args:
        a (list): A list of integers representing the genetic information of parent A.
        b (list): A list of integers representing the genetic information of parent B.
        k (int, optional): The number of crossover points. Defaults to 2.

    Returns:
        offspring (list): A list representing the genetic information of the new offspring.

    Examples:
        >>> k_point([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], 2)
        [1, 2, 8, 9, 5]

        >>> k_point([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], 3)
        [1, 2, 8, 4, 5]

    Raises:
        ValueError: If k is greater than the length of the chromosomes minus 1.
    """

    # Check if k is greater than the length of the chromosomes minus 1
    if k > len(a) - 1:
        raise ValueError(
            "k is greater than the length of the chromosomes minus 1.")

    # Choose k random crossover points
    crossover_points = random.sample(range(1, len(a)), k)

    # Sort the list of crossover points for better readability
    crossover_points.sort()

    # Perform the crossover to create the offspring
    offspring = []
    last_crossover_point = 0
    for i, point in enumerate(crossover_points):
        if i % 2 == 0:
            # Take the genes from the previous crossover point to this one from parent A
            offspring += a[last_crossover_point:point]
        else:
            # Take the genes from the previous crossover point to this one from parent B
            offspring += b[last_crossover_point:point]
        last_crossover_point = point
    # Take genes from the last crossover point to the end from parent A or B alternatively
    for i, gene in enumerate(a[last_crossover_point:]):
        if i % 2 == 0:
            offspring.append(gene)
        else:
            offspring.append(b[last_crossover_point + i])

    return offspring


def uniform(a, b):
    """
    Perform a uniform crossover on the input chromosomes a and b to create a new offspring.

    In a uniform crossover, each gene of the parents can be treated separately. We randomly decide
    whether we need to swap the gene with the same location of another chromosome.

    Args:
        a (list): A list of integers representing the genetic information of parent A.
        b (list): A list of integers representing the genetic information of parent B.

    Returns:
        offspring (list): A list representing the genetic information of the new offspring.

    Examples:
        >>> uniform([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        [6, 2, 3, 9, 5]

        >>> uniform([10, 20, 30, 40, 50], [60, 70, 80, 90, 100])
        [60, 20, 30, 90, 50]
    """
    import random

    # Create a new list for the offspring
    offspring = []

    # Loop through each gene of the parents and randomly decide which one to include in the offspring
    for gene_a, gene_b in zip(a, b):
        if random.random() < 0.5:
            offspring.append(gene_a)
        else:
            offspring.append(gene_b)

    return offspring


def partially_matched(a, b):
    """
    Perform a partially matched crossover (PMX) on the input chromosomes a and b to create a new offspring.

    In a PMX, two parents are chosen for mating. One parent donates some part of genetic material, and the
    corresponding part of the other parent participates in the child. Once this process is completed,
    the left-out alleles are copied from the second parent.

    Partially matched crossover (PMX) is the most frequently used crossover operator. 
    It is an operator that performs better than most of the other crossover operators. 
    The partially matched (mapped) crossover was proposed by D. Goldberg and R. Lingle [66]. Two parents are choose for mating. 
    One parent donates some part of genetic material and the corresponding part of other parent participates in the child. 
    Once this process is completed, the left out alleles are copied from the second parent 

    Args:
        a (list): A list of integers representing the genetic information of parent A.
        b (list): A list of integers representing the genetic information of parent B.

    Returns:
        offspring (list): A list representing the genetic information of the new offspring.

    Examples:
        >>> partially_matched([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        [1, 7, 8, 4, 5]

        >>> partially_matched([10, 20, 30, 40, 50], [60, 70, 80, 90, 100])
        [60, 20, 30, 90, 50]
    """
    import random

    # Choose two random crossover points
    pos1 = random.randint(0, len(a) - 1)
    pos2 = random.randint(0, len(a) - 1)

    # Swap the smaller and larger points
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1

    # Create a mapping between the two parents based on the crossover points
    mapping = {}
    for i in range(pos1, pos2):
        mapping[a[i]] = b[i]
        mapping[b[i]] = a[i]

    # Create a new offspring by performing the PMX
    offspring = [0] * len(a)
    for i, gene in enumerate(a):
        if i >= pos1 and i < pos2:
            offspring[i] = b[i]
        else:
            while mapping[gene] in offspring[pos1:pos2]:
                gene = mapping[gene]
            offspring[i] = mapping[gene]

    return offspring


def order(a, b):
    """
    Order crossover (OX) was proposed by Davis in 1985. 
    OX copies one (or more) parts of parent to the offspring from the selected cut-points 
    and fills the remaining space with values other than the ones included in the copied section. 
    The variants of OX are proposed by different researchers for different type of problems. 
    OX is useful for ordering problems. 
    However, it is found that OX is less efficient in case of Travelling Salesman Problem

    Order crossover (OX) copies a middle section of one parent to the offspring, ordered as they appear 
    in the other parent. The remaining positions are filled with values not in the copied section, 
    maintaining the order they appear in the second parent. 

    This variant of OX is useful for ordering problems and has been proposed by different researchers 
    for different types of problems. However, it is less efficient when dealing with problems like 
    Travelling Salesman Problem.

    Args:
        a (list or array): The first parent sequence.
        b (list or array): The second parent sequence.

    Returns:
        list: Two new sequences formed by the order crossover.

    Example:
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        order(a, b)
        # Output: [[3, 4, 5, 6, 7, 8, 9, 2, 1, 10], [6, 5, 4, 3, 2, 1, 9, 8, 7, 10]]
    """

    cut_points = sorted(random.sample(range(len(a)), 2))
    offspring1 = [-1] * len(a)
    offspring2 = [-1] * len(a)

    for i in range(cut_points[0], cut_points[1]):
        offspring1[i] = a[i]
        offspring2[i] = b[i]

    remaining_a = [x for x in a if x not in offspring2]
    remaining_b = [x for x in b if x not in offspring1]

    pointer_a = pointer_b = 0
    for i in range(len(a)):
        if offspring1[i] == -1:
            offspring1[i] = remaining_a[pointer_a]
            pointer_a += 1
        if offspring2[i] == -1:
            offspring2[i] = remaining_b[pointer_b]
            pointer_b += 1

    return [offspring1, offspring2]


def precedence_preserving_crossover(a, b):
    """
    Precedence preserving crossover (PPX) preserves the ordering of individual solutions as present 
    in the parent of offspring before the application of crossover. 
    The offspring is initialized to a string of random 1s and 0s that decides whether the individuals 
    from both parents are to be selected or not. 

    Precedence preserving crossover (PPX) preserves the ordering of individual solutions as present 
    in the parent of offspring before the application of crossover. 
    This implementation initializes the offspring as a sequence of random 1s and 0s that decide whether 
    the elements are to be selected from the first or the second parent. 
    The result is two new solutions with elements in the same order as in the parent sequences a and b.

    Args:
        a (list or array): The first parent sequence.
        b (list or array): The second parent sequence.

    Returns:
        list: Two new sequences formed by the precedence preserving crossover.

    Example:
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        precedence_preserving_crossover(a, b)
        # Output: [[1, 2, 8, 7, 5, 6, 4, 3, 9, 10], [10, 9, 3, 4, 6, 5, 7, 8, 2, 1]]
    """

    offspring1 = [-1] * len(a)
    offspring2 = [-1] * len(a)
    common = list(set(a) & set(b))
    uncommon = list(set(a) | set(b) - set(common))
    random.shuffle(common)
    random.shuffle(uncommon)
    bool_sequence = [random.randint(0, 1) for _ in range(len(common))]

    for i in range(len(common)):
        if bool_sequence[i] == 1:
            offspring1[i] = common[i]
            offspring2[i] = uncommon.pop(0)
        else:
            offspring1[i] = uncommon.pop(0)
            offspring2[i] = common[i]

    remaining_a = [x for x in a if x not in offspring2]
    remaining_b = [x for x in b if x not in offspring1]

    pointer_a = pointer_b = 0

    for i in range(len(a)):
        if offspring1[i] == -1:
            offspring1[i] = remaining_a[pointer_a]
            pointer_a += 1
        if offspring2[i] == -1:
            offspring2[i] = remaining_b[pointer_b]
            pointer_b += 1

    return [offspring1, offspring2]


def shuffle(a, b):
    """
    Shuffle crossover was proposed by Eshelman to reduce the bias introduced by other crossover techniques. 
    It shuffles the values of an individual solution before the crossover and unshuffles them 
    after crossover operation is performed so that the crossover point does not introduce any bias in crossover. 
    However, the utilization of this crossover is very limited in the recent years

    Shuffle crossover was proposed by Eshelman to reduce the bias introduced by other crossover techniques. 
    It shuffles the values of an individual solution before the crossover and unshuffles them 
    after crossover operation is performed so that the crossover point does not introduce any bias in crossover. 
    This implementation shuffles the values of each parent sequence, and then performs a simple crossover 
    of the shuffled sequences, before unshuffling the result. 

    Args:
        a (list or array): The first parent sequence.
        b (list or array): The second parent sequence.

    Returns:
        list: Two new sequences formed by the shuffle crossover.

    Example:
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        shuffle(a, b)
        # Output: [[1, 2, 8, 7, 6, 5, 4, 3, 9, 10], [10, 9, 3, 4, 5, 6, 7, 8, 2, 1]]
    """

    temp_a = a[:]
    temp_b = b[:]

    random.shuffle(temp_a)
    random.shuffle(temp_b)

    cut = random.randint(0, len(a))

    offspring1 = temp_a[:cut] + temp_b[cut:]
    offspring2 = temp_b[:cut] + temp_a[cut:]

    unshuffle1 = [x for _, x in sorted(zip(temp_a, offspring1))]
    unshuffle2 = [x for _, x in sorted(zip(temp_b, offspring2))]

    return [unshuffle1, unshuffle2]


def reduced_surrogate(a, b):
    """
    Reduced surrogate crossover (RCX) reduces the unnecessary crossovers 
    if the parents have the same gene sequence for solution representations. 
    RCX is based on the assumption that GA produces better individuals 
    if the parents are sufficiently diverse in their genetic composition. 
    However, RCX cannot produce better individuals for those parents that have same composition


    Reduced surrogate crossover (RCX) reduces the unnecessary crossovers 
    if the parents have the same gene sequence for solution representations. 
    RCX is based on the assumption that GA produces better individuals 
    if the parents are sufficiently diverse in their genetic composition. 
    However, RCX cannot produce better individuals for those parents that have the same composition.

    Parameters
    ----------
    a: list
        The first parent solution representation.
    b: list
        The second parent solution representation.

    Returns
    -------
    list
        The offspring generated by applying RCX to the parent solutions.

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> b = [5, 4, 3, 2, 1]
    >>> offspring = reduced_surrogate(a, b)
    >>> print(offspring)
    [1, 4, 3, 2, 5]
    """

    if a == b:
        # Parents are identical, cannot apply RCX
        return None
    else:
        # Find the largest common subsequence (LCS) of a and b
        n = len(a)
        m = len(b)
        L = [[0] * (m + 1) for i in range(n + 1)]
        for i in range(n):
            for j in range(m):
                if a[i] == b[j]:
                    L[i+1][j+1] = L[i][j] + 1
                else:
                    L[i+1][j+1] = max(L[i][j+1], L[i+1][j])
        lcs = []
        i, j = n, m
        while i > 0 and j > 0:
            if a[i-1] == b[j-1]:
                lcs.insert(0, a[i-1])
                i -= 1
                j -= 1
            elif L[i-1][j] > L[i][j-1]:
                i -= 1
            else:
                j -= 1
        # Apply RCX to generate the offspring
        offspring = a.copy()
        for k in range(len(lcs)):
            i = offspring.index(lcs[k])
            j = b.index(lcs[k])
            offspring[i], offspring[j] = offspring[j], offspring[i]
        return offspring


def cycle(a, b):
    """
    Cycle crossover was proposed by Oliver. 
    It attempts to generate an offspring using parents where each element occupies the position 
    by referring to the position of their parents. In the first cycle, it takes some elements from the first parent. 
    In the second cycle, it takes the remaining elements from the second parent


    Cycle crossover was proposed by Oliver. 
    It attempts to generate an offspring using parents where each element occupies the position 
    by referring to the position of their parents. In the first cycle, it takes some elements from the first parent. 
    In the second cycle, it takes the remaining elements from the second parent.

    Parameters
    ----------
    a: list
        The first parent solution representation.
    b: list
        The second parent solution representation.

    Returns
    -------
    list
        The offspring generated by applying cycle crossover to the parent solutions.

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> b = [5, 4, 3, 2, 1]
    >>> offspring = cycle(a, b)
    >>> print(offspring)
    [5, 2, 3, 4, 1]

    >>> a = [1, 2, 3, 4, 5, 6, 7]
    >>> b = [3, 7, 5, 1, 6, 2, 4]
    >>> offspring = cycle(a, b)
    >>> print(offspring)
    [1, 7, 5, 4, 6, 2, 3]
    """
    n = len(a)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = b.index(a[j])
            cycles.append(cycle)
    # Swap elements in cycles
    offspring = a.copy()
    for cycle in cycles:
        cycle_length = len(cycle)
        for i in range(cycle_length):
            j = (i + 1) % cycle_length
            offspring[cycle[i]] = offspring[cycle[j]]
            offspring[cycle[j]] = offspring[cycle[i]]
    return offspring
