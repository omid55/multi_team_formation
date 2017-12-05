# Omid55
# algorithms to solve the problem of teams of teams

# Imports
import numpy as np
from queue import PriorityQueue
import itertools as itt
import matplotlib.pyplot as plt
import sys



class DivideAndConquerAlgorithm:
    def __init__(self, problem, algorithm_speed=3):
        self.problem = problem
        self.algorithm_speed = algorithm_speed


    def solve(self):
        self.solution_teams = []
        k = self.problem.k
        n = self.problem.n
        people_list = list(range(n))
        # sorting people with increasing order of their risk taking feature
        # sorted_risks = sorted(self.problem.risk_takings[people_list])
        ordered_people = [x for _,x in sorted(zip(self.problem.risk_takings[people_list], people_list))]
        self.find_teams(ordered_people, k)
        self.solution_teams = sorted([list(sorted(team)) for team in self.solution_teams])  # not necessary (just for being similar to other solutions)
        return self.solution_teams


    # it is a recursive function divides ordered_people into 2 pieces
    #   and searches each piece for half of teams_count which is required
    def find_teams(self, ordered_people, teams_count):
        if teams_count <= 1:
            # find only one team now
            self.solution_teams.append(self.find_single_team(ordered_people))
            #self.solution_teams.append(self.find_single_team_exhaustive(ordered_people))
            return
        r = self.find_where_to_divide(ordered_people, teams_count)
        self.find_teams(ordered_people[:r], int(teams_count/2))
        self.find_teams(ordered_people[r:], int(teams_count/2))


    """Where ..."""
    def find_where_to_divide(self, ordered_people, teams_count):
        m = self.problem.m
        fitness = []
        half_required_people = int(teams_count*m/2)
        half_teams_count = int(teams_count/2)
        for i in range(half_required_people, len(ordered_people)-half_required_people+1):
            left = ordered_people[:i]
            right = ordered_people[i:]
            left_est_fitness = self.estimate_fitness(left, half_teams_count)
            right_est_fitness = self.estimate_fitness(right, half_teams_count)
            # left_est_fitness = self.estimate_fitness_fast(left, half_teams_count)
            # right_est_fitness = self.estimate_fitness_fast(right, half_teams_count)
            fitness.append((left_est_fitness + right_est_fitness)/2)
        r = half_required_people + np.argmax(fitness)
        return r


    """It estimate the fitness of probable teams_count teams in people_lists"""
    def estimate_fitness(self, people_list, teams_count):
        s = self.problem.s
        speed = self.algorithm_speed
        m = self.problem.m
        if len(people_list) < m * teams_count:
            return -999  # there is not enough people left in this people_list to create team_count teams of m people with
        if speed == 3:
            return self.approximate_skills_fast(people_list, teams_count)
        elif speed == 2:
            score1_estimate = self.approximate_skills_fast(people_list, teams_count)
        elif speed == 1:
            score1_estimate = self.approximate_skills(people_list, teams_count)
        else:
            raise ValueError('speed value is not valid. It should be one of 1,2,3 but it is ' % speed)
        score3_estimate = self.approximate_risks(people_list, teams_count)
        estimate = (1 - self.problem.alpha - self.problem.beta) * score1_estimate + self.problem.beta * score3_estimate
        return estimate


    """It only considers score1 (skills)"""
    def approximate_skills_fast(self, people_list, teams_count):
        s = self.problem.s
        skill_weight = self.problem.skill_weight
        estimate = 0
        for j in range(s):
            sorted_skills = sorted(self.problem.skills[people_list, j])
            estimate += skill_weight[j] * np.mean(sorted_skills[-2 * teams_count:])
        return estimate


    """For or every skill, we consider the score distribution of average of all pairs of individuals. We sample c1
    pairs (without replacement) and consider highest teams_count/2 scores. We run this c2 times and the average expected
    skill score is obtained.
    """
    def approximate_skills(self, people_list, teams_count):
        s = self.problem.s
        people_size = len(people_list)
        couples_size = int((people_size*people_size+people_size)/2)
        skills = self.problem.skills
        # parameter setting the constants for this function
        c1 = min(3*teams_count, couples_size)
        c2 = 100
        eps = 0.1
        skill_weight = self.problem.skill_weight
        couples_scores = np.zeros(couples_size)
        approximate_score = 0
        for skill_index in range(s):  # for every skill
            cnt = 0
            for i in range(people_size):  # for every 2 people
                for j in range(i+1, people_size):
                    couple_score = (skills[people_list[i], skill_index] +
                                    skills[people_list[j], skill_index]) / 2
                    couples_scores[cnt] = couple_score
                    cnt += 1
            bin_count = min(len(couples_scores), 50)
            hist, bins = np.histogram(couples_scores, bins=bin_count)
            hist = hist + eps
            hist_probability = hist / sum(hist)
            mid_bins = bins[:-1] + np.diff(bins) / 2
            score = 0
            for run in range(c2):
                samples = np.random.choice(mid_bins, size=c1, replace=False, p=hist_probability)
                samples_sorted = sorted(samples)
                score += np.mean(samples_sorted[-teams_count:])
            score /= c2
            approximate_score += skill_weight[skill_index] * score
        return approximate_score


    def approximate_risks(self, people_list, teams_count):
        m = self.problem.m
        # parameter setting the constants for this function
        s3_estimate = 0
        c3 = 100
        for i in range(c3):
            samples = np.random.choice(people_list, size=m*teams_count, replace=False)
            teams = []
            for j in range(teams_count):
                teams.append(list(sorted(samples[j * m:(j + 1) * m])))
            s3_estimate += self.problem.score3(teams, number_of_teams=teams_count)
        s3_estimate /= c3
        return s3_estimate


    # This choice is made based on the skills of the individuals and the uniformity of network connectivity.
    def find_single_team(self, people):
        MAX_ITERATIONS = 1000
        m = self.problem.m
        alpha = self.problem.alpha
        beta = self.problem.beta
        l = len(people)
        # estimate_of_score1 = np.mean(sorted(self.problem.skills)[-2:])
        pq = PriorityQueue()    # priority queue of people in subset of teams
        for i in range(l):
            for j in range(i+1, l):
                item = [people[i], people[j]]
                estimate_score = self.problem.score1_single_team(item) # it does not have score2 because there is only one distance
                pq.put((-estimate_score, item))
        for iteration in range(MAX_ITERATIONS):   # instead of while 1
            (_, heap_top) = pq.get()
            if len(heap_top) == m:   # if it has already the size of intended team size
                return heap_top
            max_estimate = -sys.maxsize
            new_item = None
            for i in range(l):
                new_member = people[i]
                if new_member not in heap_top:
                    x = heap_top.copy()
                    x.append(new_member)
                    estimate_score_tmp = (1 - alpha - beta) * self.problem.score1_single_team(x) \
                                         + alpha * self.problem.score2_single_team(x)   # CHECK HERE << NOT EFFICIENT LIKE THIS >>
                    if estimate_score_tmp > max_estimate:
                        new_item = x
                        max_estimate = estimate_score_tmp
            pq.put((-max_estimate, new_item))
        print('It didn\'t find any single team in function "find_single_team".')
        return -5  # which means ended without finding any good team


    def find_single_team_exhaustive(self, people):
        m = self.problem.m
        if len(people) < m:
            raise ValueError('Size of people is smaller than team size %d < %d' % (len(people),m))
        # elif len(people) == m:
        #     return people
        best_score = -sys.maxsize
        best_team = None
        for team in list(itt.combinations(people, m)):
            score = (self.problem.score1_single_team(team) + self.problem.score2_single_team(team)) / 2
            if score > best_score:
                best_score = score
                best_team = team
        return best_team



class ByRandom:
    def __init__(self, problem):
        self.problem = problem


    """Randomly pick teams"""
    def solve(self):
        n = self.problem.n
        m = self.problem.m
        k = self.problem.k
        people = np.random.choice(n, m*k, replace=False)
        teams = []
        for i in range(k):
            teams.append(list(sorted(people[i*m:(i+1)*m])))
        teams = sorted(teams)
        return teams


class ByOptimal:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, given_objective_function=None):
        if given_objective_function is None:
            given_objective_function = self.problem.objective_function
        k = self.problem.k
        n = self.problem.n
        self.all_teams = []
        self._generate_all_teams(np.array(range(0, n)), k, [])
        optimal_fitness = -sys.maxsize
        optimal_team = None
        for team in self.all_teams:
            fitness = given_objective_function(team)
            if fitness > optimal_fitness:
                optimal_fitness = fitness
                optimal_team = team
        optimal_team = sorted([list(sorted(team)) for team in optimal_team])  # not necessary, only for similarity with other solutions order
        return optimal_team

    """G"""
    def _generate_all_teams(self, people, number_of_teams, teams_out):
        m = self.problem.m
        if number_of_teams <= 0 or len(people) < number_of_teams*m:
            return
        for team in list(itt.combinations(people, m)):
            team_list = sorted(team)
            teams_out.append(team_list)
            if number_of_teams == 1:
                self.all_teams.append(teams_out.copy())
            else:
                # deleting team and members before its smallest value
                #   first, finding team indices
                to_be_deleted_indices = []
                for i in range(len(team_list)):
                    index = np.where(people == team_list[i])[0]
                    if not i: # the first one
                        to_be_deleted_indices.extend(list(range(index[0]+1)))
                    else:
                        to_be_deleted_indices.extend(index)
                rest_of_people = np.delete(people, to_be_deleted_indices)
                if len(rest_of_people) >= (number_of_teams-1)*m:
                    self._generate_all_teams(rest_of_people, number_of_teams - 1, teams_out)
            teams_out.remove(team_list)


"""Evolutionary algorithms (non-convex optimizers)"""

"""Randomly initializing the population of genomes"""
def init_population(problem, population_size):
    n = problem.n
    m = problem.m
    k = problem.k
    population = []
    for i in range(population_size):
        people = np.random.choice(n, m * k, replace=False)
        chromosome = np.zeros(n)
        for j in range(k):
            index = list(sorted(people[j * m:(j + 1) * m]))
            chromosome[index] = j+1
        population.append(chromosome)
    return population


"""Computing fitness of all genomes in the population"""
def compute_fitness_for_all(problem, population):
    # computing their fitnesses
    fitnesses = []
    for chromosome in population:
        fitnesses.append(problem.objective_function(convert_chromosome_to_teams(chromosome)))
    fitnesses = np.array(fitnesses)
    return fitnesses


def convert_chromosome_to_teams(chromosome):
    teams = []
    for v in np.unique(chromosome):
        if v > 0:
            indices = np.where(chromosome == v)[0]
            teams.append(indices)
    return teams


def plot_fitness_figure(obj, name):
    plt.title(name + ': fitness dynamics figure')
    plt.plot(obj.fitness_means)
    plt.plot(obj.fitness_maxs)
    plt.legend(['Mean', 'Max'])
    plt.show()


# ---------------------------- GA ---------------------------------
"""Genetic Algorithm as a non-convex optimizer"""
class GeneticAlgorithm:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, settings=None):
        if not settings:
            settings = {'population_size': 30, 'crossover_prob': 0.8, 'mutation_prob': 0.3,\
                        'max_iteration': 1000, 'gene_permutation_portion': 0.2, 'eps': 0.0001,\
                        'with_figure': True}
        # unstructing the settings
        population_size = settings['population_size']
        crossover_prob = settings['crossover_prob']
        mutation_prob = settings['mutation_prob']
        max_iteration = settings['max_iteration']
        gene_permutation_portion = settings['gene_permutation_portion']
        eps = settings['eps']
        with_figure = settings['with_figure']

        # initialization
        population = init_population(self.problem, population_size)

        # computing fitnesses
        fitnesses = compute_fitness_for_all(self.problem, population)
        self.fitness_maxs = [np.max(fitnesses)]
        self.fitness_means = [np.mean(fitnesses)]

        for iteration in range(max_iteration):

            # plot the fitness changes along the way
            if with_figure:
                plt.plot(self.fitness_means)
                plt.plot(self.fitness_maxs)
                plt.legend(['Mean', 'Max'])
                plt.draw()
                plt.pause(0.01)

            # check if convergence has happened already
            if abs(self.fitness_maxs[-1] - self.fitness_means[-1]) < eps:
                break

            # cross over
            children = self.crossover(population, fitnesses, crossover_prob)

            # mutation
            mutated_children = self.mutate(children, mutation_prob, gene_permutation_portion)

            # the new generation
            new_generation = mutated_children
            new_generation.extend(children)

            # # OPTIONAL CHECK
            # self.check_this_population(new_generation)

            # adding new generation and theirs fitness to rest of population and fitnesses
            new_fitnesses = compute_fitness_for_all(self.problem, new_generation)
            population.extend(new_generation)
            fitnesses = np.concatenate((fitnesses, new_fitnesses))

            # survival selection
            population, fitnesses = self.survival_selection(population, fitnesses, population_size)

            # adding to mean and max fitnesses list
            self.fitness_maxs.append(np.max(fitnesses))
            self.fitness_means.append(np.mean(fitnesses))

        # picking the best chromosome and creating best team from that
        best_team_chromosome = population[-1]  # the last one is the one with the best fitness
        best_teams = convert_chromosome_to_teams(best_team_chromosome)
        plt.close()
        best_teams = sorted([list(sorted(team)) for team in
                             best_teams])  # not necessary, only for similarity with other solutions order
        return best_teams


    """Performing the cross over function on population"""
    def uniform_crossover(self, population, fitnesses, crossover_prob):
        return NotImplementedError  # NOT IMPLEMENTED YET << CHECK HERE >>

    def crossover(self, population, fitnesses, crossover_prob):
        population_size = len(population)
        if population_size != len(fitnesses):
            raise ValueError('Lengths of population and fitnesses are not same len(population): %d != len(fitnesses): %d' (population_size, len(fitnesses)))
        k = self.problem.k
        L = len(population[0])
        crossovered_population = []
        # choosing for crossover based on their fitnesses
        eps = 0.01   # this is because we don't want all of them become 0, consider fitn = [0.2, 0.2, 0.2, 0.2, 0.2], then all become 0.
        fitnesses -= (min(fitnesses) - eps)
        fitnesses_as_distribution = fitnesses / np.sum(fitnesses)
        for i in range(int(population_size/2)):
            if np.random.rand() <= crossover_prob:
                selected = np.random.choice(range(len(population)), 2, replace=False, p=fitnesses_as_distribution)
                a = population[selected[0]]
                b = population[selected[1]].copy()
                b[np.where(b > 0)[0]] += k
                # 2-point crossover
                break_point = 1 + np.random.randint(L-2)   # an index between [1, L-2]
                child1 = np.zeros(L)
                child2 = np.zeros(L)
                child1[:break_point] = a[:break_point]
                child1[break_point:] = b[break_point:]
                child2[:break_point] = b[:break_point]
                child2[break_point:] = a[break_point:]
                child1 = self.fix_chromosome(child1)
                child2 = self.fix_chromosome(child2)
                crossovered_population.append(child1)
                crossovered_population.append(child2)
        return crossovered_population


    """Fixing a child chromosome from mistakes in the number of teams and teammate after crossover"""
    # for testing try:
    #   with n = 9, m = 4, k = 2
    # print(self.fix_chromosome(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])))
    # print(self.fix_chromosome(np.array([1, 1, 1, 1, 2, 2, 2, 2, 2])))
    # print(self.fix_chromosome(np.array([0, 0, 0, 0, 0, 1, 0, 0, 2])))
    # print(self.fix_chromosome(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])))
    def fix_chromosome(self, chromosome):
        k = self.problem.k
        m = self.problem.m
        # keep the bigger teams and remove smaller ones
        vals, counts = np.unique(chromosome, return_counts=True)
        # we remove 0 because we don't care about it
        if not vals[0]:
            vals = vals[1:]
            counts = counts[1:]
        ordered_nonzero_vals = [x for _,x in sorted(zip(-counts,vals))]
        if len(ordered_nonzero_vals) > k:
            to_be_deleted_vals = ordered_nonzero_vals[k:]
            for v in to_be_deleted_vals:
                indices = np.where(chromosome == v)[0]
                chromosome[indices] = 0
            ordered_nonzero_vals = ordered_nonzero_vals[:k]
        team_count_so_far = 0
        for v in ordered_nonzero_vals:
            indices = np.where(chromosome == v)[0]
            if team_count_so_far >= k:
                chromosome[indices] = 0
                continue
            if len(indices) > m:
                to_be_deleted = np.random.choice(indices, len(indices)-m, replace=False)
                chromosome[to_be_deleted] = 0
            elif len(indices) < m:
                zero_indices = np.where(chromosome == 0)[0]
                to_be_added = np.random.choice(zero_indices, m - len(indices), replace=False)
                chromosome[to_be_added] = v
            team_count_so_far += 1
        if team_count_so_far < k:
            more_teams = k-team_count_so_far
            zero_indices = np.where(chromosome == 0)[0]
            people = np.random.choice(zero_indices, m * more_teams, replace=False)
            maxvalue = np.max(chromosome)
            for j in range(more_teams):
                index = list(sorted(people[j * m:(j + 1) * m]))
                chromosome[index] = maxvalue + j + 1
        # if they have larger values than k, convert them to [1,k]
        for i, v in enumerate(np.unique(chromosome)):
            if v > 0 and v != i:
                indices = np.where(chromosome == v)[0]
                chromosome[indices] = i
        return chromosome


    """Performing the mutation technique on population with probability permutation_prob"""
    def mutate(self, population, permutation_prob, gene_permutation_portion):
        population_size = len(population)
        L = len(population[0])
        mutated_population = []
        for i in range(population_size):
            if np.random.rand() <= permutation_prob:
                gene_permutation_times = int(L * gene_permutation_portion / 2)
                chromosome = population[i].copy()
                for i in range(gene_permutation_times):
                    selected = np.random.choice(range(L), 2, replace=False)           # choose 2 genes by random
                    i1 = selected[0]
                    i2 = selected[1]
                    chromosome[i1], chromosome[i2] = chromosome[i2], chromosome[i1]   # and simply swap 2 genes
                mutated_population.append(chromosome)
        return mutated_population


    """Picking the best chromosomes to last to the next generation"""
    def survival_selection(self, population, fitnesses, population_size):
        # return [x for _,x in sorted(zip(fitnesses,population))][-population_size:], sorted(fitnesses)[-population_size:]
        sorted_population_indices = np.array([x for _, x in sorted(zip(fitnesses, range(len(population))))])  # based on fitness
        new_population_indices = sorted_population_indices[-population_size:]                                 # elitism: best of the population
        sorted_fitnesses = sorted(fitnesses)
        new_fitnesses = sorted_fitnesses[-population_size:]
        new_population = [population[i] for i in new_population_indices]
        return new_population, new_fitnesses


    """(OPTIONAL) Checking the population of the new generation if they are all valid"""
    def check_this_population(self, new_generation):
        k = self.problem.k
        m = self.problem.m
        n = self.problem.n
        for chromosome in new_generation:
            if len(chromosome) != n:
                print('ERROR1')
            self.check_this_chromosome(chromosome)

    def check_this_chromosome(self, chromosome):
        k = self.problem.k
        m = self.problem.m
        n = self.problem.n
        for i in range(1,k):
            if len(np.where(chromosome == i)[0]) != m:
                print('ERROR2')
                return -1
        return 0


# ---------------------------- DE ---------------------------------
"""Differential Evolution (DE) as a non-convex optimizer"""
class DifferentialEvolution:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, settings=None):
        if not settings:
            settings = {'population_size': 30,\
                        'max_iteration': 1000, 'eps': 0.0001,\
                        'beta': 0.8, 'Pr': 0.6, 'Nv': 1,\
                        'coef': 0.7, 'with_figure': True}
        # unstructing the settings
        population_size = settings['population_size']
        max_iteration = settings['max_iteration']
        eps = settings['eps']
        beta = settings['beta']
        Pr = settings['Pr']
        Nv = settings['Nv']
        coef = settings['coef']
        with_figure = settings['with_figure']

        # initialization
        population = init_population(self.problem, population_size)

        # computing fitnesses
        fitnesses = compute_fitness_for_all(self.problem, population)
        self.fitness_maxs = [np.max(fitnesses)]
        self.fitness_means = [np.mean(fitnesses)]

        for iteration in range(max_iteration):

            # plot the fitness changes along the way
            if with_figure:
                plt.plot(self.fitness_means)
                plt.plot(self.fitness_maxs)
                plt.legend(['Mean', 'Max'])
                plt.draw()
                plt.pause(0.01)

            # check if convergence has happened already
            if abs(self.fitness_maxs[-1] - self.fitness_means[-1]) < eps:
                break

            diff = len(np.where(fitnesses <= np.mean(fitnesses))[0])
            if diff >= coef * population_size:
                break

            # creating trial vector and crossover
            population, fitnesses = self.CreateTrialVectorAndCrossOver(population, fitnesses, beta, Pr, Nv)

            # adding to mean and max fitnesses list
            self.fitness_maxs.append(np.max(fitnesses))
            self.fitness_means.append(np.mean(fitnesses))

        # picking the best chromosome and creating best team from that
        best_index = np.where(fitnesses == np.max(fitnesses))[0][0]
        best_team_chromosome = population[best_index]
        best_teams = convert_chromosome_to_teams(best_team_chromosome)
        plt.close()
        best_teams = sorted([list(sorted(team)) for team in
                            best_teams])  # not necessary, only for similarity with other solutions order
        return best_teams


    def CreateTrialVectorAndCrossOver(self, population, fitnesses, Beta, Pr, Nv):
        # REMOVE IT
        TEMP = GeneticAlgorithm(problem=self.problem)
        # REMOVE IT
        population_size = len(population)
        genesNum = len(population[0])
        k = self.problem.k
        next_generation = []
        fitness_of_next_generation = []

        for i in range(population_size):
            Xi = population[i]
            fXi = fitnesses[i]

            if Nv == 1:
                # picking 3 samples randomly to create the trial vector
                idx = np.random.choice(population_size, 3, replace=False)
                x1 = population[idx[0]]
                x2 = population[idx[1]].copy()
                x2[np.where(x2 > 0)[0]] += k
                x3 = population[idx[2]].copy()
                x3[np.where(x3 > 0)[0]] += 2*k
                Ui = x1 + Beta * (x2 - x3)
            else:
                print('NOT IMPLEMENTED FOR LARGER NVs.')

            # applying Binomial Crossover
            # selecting a set of indices and adding jStar to only at least have one selected always
            jStar = np.random.randint(genesNum)
            J = set(np.where(np.random.rand(genesNum) < Pr)[0])
            J.add(jStar)
            J = list(J)

            # creating the child
            child = Xi.copy()
            child[J] = Ui[J]

            # fixing the child
            child = np.round(child)
            child -= (min(child)-1)
            child = TEMP.fix_chromosome(child)

            # deciding which one is going to the next generation
            child_fitness = self.problem.objective_function(convert_chromosome_to_teams(child))
            if child_fitness > fXi:
                next_generation.append(child)
                fitness_of_next_generation.append(child_fitness)
            else:
                next_generation.append(Xi)
                fitness_of_next_generation.append(fXi)

        return next_generation, fitness_of_next_generation

