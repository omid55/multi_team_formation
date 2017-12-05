# Omid55
import teams_of_teams_problem
import numpy as np
import time
import matplotlib.pyplot as plt
import methods


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
    objs = [problem.score1, problem.score2, problem.score3, problem.objective_function]
    for obj in objs:
        print(obj)
        start_time = time.time()
        fitnesses = np.zeros(population_size)
        for i, instance in enumerate(population):
            fitnesses[i] = obj(instance)
        duration = round(time.time() - start_time, 2)
        print('min:', min(fitnesses), ', max:', max(fitnesses), \
              ', mean:', np.mean(fitnesses), ', std:', np.std(fitnesses))
        print('(in ', duration, 's).')
        print('Linear transformation: (x-', min(fitnesses), ')*', 1 / (max(fitnesses) - min(fitnesses)))
        # plotting
        #sns.distplot(fitnesses)
        plt.hist(fitnesses)
        plt.show()
        print('\n')


def sample_for_score3():
    prob = teams_of_teams_problem.Problem(n=8, m=2, k=3, s=1)
    prob.risk_takings = np.array([0, 0.2, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    # prob.risk_takings = np.array([0.1, 0.1, 0.3, 0.5, 0.9, 0.9, 0.2, 0.5])
    # prob.risk_takings = np.array([0, 0, 0.3, 0.5, 1, 1, 0.2, 0.5])
    print(prob.score3([[0, 1], [4, 5], [6, 7]]))
    print(prob.score3([[2, 3], [4, 5], [6, 7]]))


def sample_for_score1():
    prob = teams_of_teams_problem.Problem(n=7, m=3, k=1, s=2)
    # prob.skills = np.array([[0.5,0.1], [0.2,0.3], [0.1,0.5], [0.5,0.1], [0.3,0.4]])
    prob.skills = np.array([[0.5, 0.1], [0.2, 0.3], [0.1, 0.5], [0.3, 0.3], [0.3, 0.4], [0.5, 0.2], [0.45, 0.15]])
    byoptimal = methods.ByOptimal(prob)
    optimal_team = byoptimal.solve(prob.score1)
    print(optimal_team)
    print('Optimal:\t\t', prob.score1(optimal_team))


def main(RESCALE=False, DISTRIBUTION='Uniform'):
    prob = teams_of_teams_problem.Problem(n=1000, m=20, k=16, s=8, alpha=1/3, beta=1/3, RESCALE=RESCALE, DISTRIBUTION=DISTRIBUTION)
    plot_all_for_problem(prob)


if __name__ == '__main__':
    # main()
    # sample_for_score3()
    sample_for_score1()

