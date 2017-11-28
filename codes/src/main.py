import teams_of_teams_problem
import methods
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all_for_problem(problem):
    n = problem.n
    m = problem.m
    k = problem.k
    # a distribution of all score functions
    # creating a population
    population_size = 1000
    population = []
    for i in range(population_size):
        people = np.random.choice(n, m * k, replace=False)
        instance = [list(sorted(people[i * m:(i + 1) * m])) for i in range(k)]
        population.append(instance)
    # computing their fitnesses
    # objectives
    objs = [problem.score1, problem.score2, problem.score3, problem.objective_function]
    for obj in objs:
        print(obj)
        start_time = time.time()
        fitnesses = []
        for instance in population:
            fitnesses.append(obj(instance))
        fitnesses = np.array(fitnesses)
        duration = round(time.time() - start_time, 2)
        print('(in ', duration, 's).')
        # plotting
        #sns.distplot(fitnesses)
        print('min: ', min(fitnesses), '\nmax: ' ,max(fitnesses),'\n')
        plt.hist(fitnesses)
        plt.show()
        print('\n\n\n\n')



def main():
    # parameters begins
    # n = 1000
    # m = 20
    # k = 16

    # n = 10
    # m = 3
    # k = 2

    n = 7
    m = 3
    k = 2

    #n = 7
    # n = 5
    # m = 2
    # k = 2

    s = 1

    alpha = 1/3
    beta = 1/3
    skill_weight = np.ones(s) / s  # all skills are equally important
    # parameters ends
    problem = teams_of_teams_problem.Problem(n=n, s=s, m=m, k=k, alpha=alpha, beta=beta, skill_weight=skill_weight, ALWAYS_SAME=True)

    # plot_all_for_problem(problem)


    # Solutions:
    print('Solutions:\n')

    # Random
    start_time = time.time()
    byrandom = methods.ByRandom(problem)
    random_team = byrandom.solve()
    print('Random:\t\t\t\t', problem.objective_function(random_team)) # print('Random:\t\t\t', problem.objective_function(random_team), random_team)
    duration = round(time.time() - start_time, 2)
    print('Valid: ', problem.validate_solution(random_team))
    print('(in ', duration, 's).\n')

    # Dvidie & Conquer
    start_time = time.time()
    byDAC = methods.DivideAndConquerAlgorithm(problem)
    solution_teams = byDAC.solve()
    print('Divide&Conquer:\t', problem.objective_function(solution_teams))
    duration = round(time.time() - start_time, 2)
    print('Valid: ', problem.validate_solution(solution_teams))
    print('(in ', duration, 's).\n')
    del(byDAC)
    gc.collect()

    # Differential Evolution (DE)
    start_time = time.time()
    byDE = methods.DifferentialEvolution(problem)
    de_team = byDE.solve(settings = {'population_size': 100,\
                        'max_iteration': 100, 'eps': 0.0001,\
                        'beta': 0.8, 'Pr': 0.9, 'Nv': 1,\
                        'coef': 0.7, 'with_figure': False})
    print('DE:\t\t\t\t', problem.objective_function(de_team))
    duration = round(time.time() - start_time, 2)
    print('Valid: ', problem.validate_solution(de_team))
    print('(in ', duration, 's).\n')
    gc.collect()

    # Genetic Algorithm (GA)
    start_time = time.time()
    byGA = methods.GeneticAlgorithm(problem)
    ga_team = byGA.solve(settings = {'population_size': 100, 'crossover_prob': 0.8, 'mutation_prob': 0.3,\
                        'max_iteration': 100, 'gene_permutation_portion': 0.2, 'eps': 0.0001,\
                        'with_figure': False})
    print('GA:\t\t\t\t', problem.objective_function(ga_team))
    duration = round(time.time() - start_time, 2)
    print('Valid: ', problem.validate_solution(ga_team))
    print('(in ', duration, 's).\n')
    # del(byGA)
    gc.collect()

    # Optimal (Exhaustive Search)
    if n <= 10:
        start_time = time.time()
        byoptimal = methods.ByOptimal(problem)
        optimal_team = byoptimal.solve()
        print('Optimal:\t\t', problem.objective_function(optimal_team))
        duration = round(time.time() - start_time, 2)
        print('(in ', duration, 's).\n')
        del(byoptimal)
        gc.collect()

    # showing all figures now
    methods.plot_fitness_figure(byGA, 'GA')
    methods.plot_fitness_figure(byDE, 'DE')



if __name__ == '__main__':
    main()
