import teams_of_teams_problem
import methods
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython.display import display, HTML


def main_all(settings='small'):
    if settings == 'small':
        parameters = [[5, 2, 2, 1], [7, 2, 2, 1], [7, 3, 2, 1], [9, 2, 4, 1]]
        runs = 30
    else:
        parameters = [[100, 5, 4, 1], [500, 10, 8, 4], [1000, 20, 16, 8]]
        runs = 10
    times = []
    for params in parameters:
        print(params)
        objectives = []
        for i in range(runs):
            problem = teams_of_teams_problem.Problem(n=params[0], m=params[1], k=params[2], s=params[3], alpha=1/3, beta=1/3)

            start_time = time.time()
            byrandom = methods.ByRandom(problem)
            objectives.append(['Random', problem.objective_function(byrandom.solve())])
            times.append(['Random', round(time.time() - start_time, 2)])

            start_time = time.time()
            dac = methods.DivideAndConquerAlgorithm(problem)
            objectives.append(['Proposed', problem.objective_function(dac.solve())])
            times.append(['Proposed', round(time.time() - start_time, 2)])

            start_time = time.time()
            byDE = methods.DifferentialEvolution(problem)
            objectives.append(['DE', problem.objective_function(byDE.solve(\
                settings = {'population_size': 50,\
                        'max_iteration': 100, 'eps': 0.0001,\
                        'beta': 0.8, 'Pr': 0.6, 'Nv': 1,\
                        'coef': 0.7, 'with_figure': False}))])
            times.append(['DE', round(time.time() - start_time, 2)])

            start_time = time.time()
            byGA = methods.GeneticAlgorithm(problem)
            objectives.append(['GA', problem.objective_function(byGA.solve(\
                settings = {'population_size': 50, 'crossover_prob': 0.8, 'mutation_prob': 0.3,\
                        'max_iteration': 100, 'gene_permutation_portion': 0.2, 'eps': 0.0001,\
                        'with_figure': False}))])
            times.append(['GA', round(time.time() - start_time, 2)])

            if problem.n <= 10:
                start_time = time.time()
                byoptimal = methods.ByOptimal(problem)
                objectives.append(['Optimal', problem.objective_function(byoptimal.solve())])
                times.append(['Optimal', round(time.time() - start_time, 2)])
        # fitnesses:
        objectives_df = pd.DataFrame(objectives, columns=['method', 'fitness'])
        sns.violinplot(x="method", y="fitness", data=objectives_df)
        plt.show()
        # duration times:
        #   figure:
        times_df = pd.DataFrame(times, columns=['method', 'duration (s)'])
        sns.boxplot(x="method", y="duration (s)", data=times_df)
        plt.show()
        #   table:
        times_df.sort_values('method', inplace=True)
        dt = []
        for name, group in times_df.groupby('method'):
            dt.append([name, str(round(group['duration (s)'].mean(), 2)) +
                       '+-' + str(round(group['duration (s)'].std(), 2))])
        times_stats_df = pd.DataFrame(dt, columns=['name', 'duration (s)'])
        display(times_stats_df)
        print('\n\n')


if __name__ == '__main__':
    main_all()