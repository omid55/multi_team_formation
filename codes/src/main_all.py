import teams_of_teams_problem
import methods
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def main_all():
    parameters = [[5, 2, 2, 1], [7, 2, 2, 1], [7, 3, 2, 1], [9, 2, 4, 1]]
    runs = 30
    for params in parameters:
        objectives = []
        for i in range(runs):
            problem = teams_of_teams_problem.Problem(n=params[0], m=params[1], k=params[2], s=params[3], alpha=1/3, beta=1/3)

            byrandom = methods.ByRandom(problem)
            objectives.append(['Random', problem.objective_function(byrandom.solve())])

            dac = methods.DivideAndConquerAlgorithm(problem)
            objectives.append(['DAC', problem.objective_function(dac.solve())])

            byDE = methods.DifferentialEvolution(problem)
            objectives.append(['DE', problem.objective_function(byDE.solve(\
                settings = {'population_size': 50,\
                        'max_iteration': 100, 'eps': 0.0001,\
                        'beta': 0.8, 'Pr': 0.6, 'Nv': 1,\
                        'coef': 0.7, 'with_figure': False}))])

            byGA = methods.GeneticAlgorithm(problem)
            objectives.append(['GA', problem.objective_function(byGA.solve(\
                settings = {'population_size': 50, 'crossover_prob': 0.8, 'mutation_prob': 0.3,\
                        'max_iteration': 100, 'gene_permutation_portion': 0.2, 'eps': 0.0001,\
                        'with_figure': False}))])

            byoptimal = methods.ByOptimal(problem)
            objectives.append(['Optimal', problem.objective_function(byoptimal.solve())])
        objectives_df = pd.DataFrame(objectives, columns=['method', 'fitness'])
        sns.violinplot(x="method", y="fitness", data=objectives_df)
        plt.show()

if __name__ == '__main__':
    main_all()