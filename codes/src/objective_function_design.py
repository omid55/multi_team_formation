# Omid55
import teams_of_teams_problem
import numpy as np
import time
import matplotlib.pyplot as plt


def analyze_the_objective_function(problem, objective_function, population_size=10000):
    n = problem.n
    m = problem.m
    k = problem.k
    population = []
    for i in range(population_size):
        people = np.random.choice(n, m * k, replace=False)
        instance = [list(sorted(people[i * m:(i + 1) * m])) for i in range(k)]
        population.append(instance)
    # computing their fitnesses
    fitnesses = []
    for instance in population:
        fitnesses.append(objective_function(instance))



def plot_all_for_problem(problem):
    n = problem.n
    m = problem.m
    k = problem.k
    # a distribution of all score functions
    # creating a population
    population_size = 10000
    population = []
    for i in range(population_size):
        people = np.random.choice(n, m * k, replace=False)
        instance = [list(sorted(people[i * m:(i + 1) * m])) for i in range(k)]
        population.append(instance)
    # computing their fitnesses
    # objectives
    objs = [problem.score1, problem.score2, problem.score3_meanstd, problem.score3, problem.objective_function]
    # objs = [problem.score1]
    for obj in objs:
        print(obj)
        start_time = time.time()
        fitnesses = []
        for instance in population:
            fitnesses.append(obj(instance))
        fitnesses = np.array(fitnesses)
        duration = round(time.time() - start_time, 2)
        print('min:', min(fitnesses), ', max:', max(fitnesses), \
              ', mean:', np.mean(fitnesses), ', std:', np.std(fitnesses))
        print('(in ', duration, 's).')
        # plotting
        #sns.distplot(fitnesses)
        plt.hist(fitnesses)
        plt.show()
        print('\n')


if __name__ == '__main__':
    prob = teams_of_teams_problem.Problem(n=1000, s=1, m=10, k=16, alpha=1/3, beta=1/3)
    plot_all_for_problem(prob)