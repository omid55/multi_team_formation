# Omid55
# teams of teams problem data generation
# it includes the generation part plus the objective function

# imports
import numpy as np
import networkx as nx
import myutils
import time

# n: number of individuals
# m: number of team members
# t: number of teams
# s: number of skills
class Problem:
    def __init__(self, n, s, m, t, alpha=1/3, beta=1/3, skill_weight=None, ALWAYS_SAME=False, RESCALE=True, DISTRIBUTION='Uniform'):
        self.n = n
        self.m = m
        self.t = t
        self.s = s
        self.alpha = alpha
        self.beta = beta
        if skill_weight is None:   # which means not defined
            # we consider all skills are equally important
            self.skill_weight = np.ones(s) / s   # this means all of skills have same impact in success
        elif np.sum(skill_weight) == 1 and len(skill_weight) == s:
            self.skill_weight = skill_weight
        else:
            raise ValueError('Error in skill_weight parameter: Not expected structure/values.')
        self.ALWAYS_SAME = ALWAYS_SAME
        self.RESCALE = RESCALE
        self.DISTRIBUTION = DISTRIBUTION
        self._generate()
        if self.RESCALE:
            self.compute_scores_boundaries()
        # **kwargs
        # for key, value in kwargs.iteritems():
        #     setattr(self, key, value)


    """It generates a dataset of people
    with different values for skills, risk taking and
    network distance.
    n: number of people
    s: number of skills (skill dimensions)
    """
    def _generate(self):
        if self.ALWAYS_SAME:
            np.random.seed(1)  # to make all random values same
        # skills of every person: every person has a list of s elements describing his/her skill values fall in [0-1].
        if self.DISTRIBUTION == 'Uniform':
            self.skills = np.random.rand(self.n, self.s)
        elif self.DISTRIBUTION == 'Normal':
            self.skills = np.random.randn(self.n, self.s) * 0.1666 + 0.5
        else:
            raise ValueError('ERROR in DISTRIBUTION value: ' % self.DISTRIBUTION)
        # risk taking of every person: it is also a value between [0,1].
        if self.DISTRIBUTION == 'Uniform':
            self.risk_takings = np.random.rand(self.n)
        elif self.DISTRIBUTION == 'Normal':
            self.risk_takings = np.random.randn(self.n) * 0.1666 + 0.5
        # network connectivity (or distance) is a matrix of n*n: we always can for any
        #   graph, compute the shortest path among all nodes
        #   and normalize it between [0,1]. We assume, relationships are not directed.
        if self.ALWAYS_SAME:
            # it keeps the same numbers which is helpful for debugging purposes
            G = nx.to_undirected(nx.scale_free_graph(self.n, seed=1))
        else:
            # it changes the structure every time we run
            G = nx.to_undirected(nx.scale_free_graph(self.n))
        D = nx.floyd_warshall_numpy(G)
        D /= np.max(D)
        self.network_connectivity = D

    def get_all_objectives(self, teams):
        return [self.objective_function(teams), self.score1(teams), self.score2(teams), self.score3(teams)]

    """This function computes an empirical min and max estimate for each score in order to normalize them automatically,
        since we want them to fall in [0,1] each to have no priority to each other
        params::
        empirical_population_size: the size of random sample population we generate to get a rough estimate of 
        distribution (increase to get more accurate estimates)
        eps: the small amount of error we consider to increase empirical maximum and decrease empirical minimum to get 
        the real max and min, respectively
        (empirical_population_size=1000, eps=0.01)"""
    def compute_scores_boundaries(self, empirical_population_size=1000, eps=0.01, verbose=False):
        m = self.m
        n = self.n
        t = self.t
        start_time = time.time()
        population = []
        for i in range(empirical_population_size):
            people = np.random.choice(n, m * t, replace=False)
            instance = [list(sorted(people[i * m:(i + 1) * m])) for i in range(t)]
            population.append(instance)
        # computing their fitnesses
        # all objective functions
        objectives = [self.score1, self.score2, self.score3]
        objective_names = ['score1', 'score2', 'score3']
        self.boundaries = {}
        for objective_index, objective in enumerate(objectives):
            fitnesses = np.zeros(empirical_population_size)
            for i, instance in enumerate(population):
                fitnesses[i] = objective(instance)
            self.boundaries[objective_names[objective_index]] = {'min': min(fitnesses)-eps, 'max': max(fitnesses)+eps}
            del fitnesses
        del population[:]
        if verbose:
            duration = round(time.time() - start_time, 2)
            print('Computing boundaries was done in: ', duration, 's.\n')

    # teams: is a list of teams which each team is a list of indices (individual indices)
    #   for instance: teams = [[1,2,4],[0,3,5]]
    #   n: #individuals
    #   t: #teams
    #   m: size of each team
    def objective_function(self, teams):
        # score based on 3 different factors
        score = (1 - self.alpha - self.beta) * self.score1(teams) + self.alpha * self.score2(teams)\
                + self.beta * self.score3(teams)
              # + self.beta * self.score3_meanstd(teams)
        return score

    # score1 is defined as average value of score1 on each individual member
    # it returns a number in [0,1]
    def score1(self, teams):
        s1 = 0
        for team in teams:
            s1 += self.score1_single_team(team)
        s1 /= len(teams)
        return s1

    # let avg2(t,j) be the average of top-2 skill values of the individuals in the team t for skill j.
    #   For team t, score1_single_team(t) =(1/s)*sum(avg2(t,j))
    # it returns a number in [0,1]
    def score1_single_team(self, team):
        s1 = 0
        for i in range(self.s):
            s1 += self.skill_weight[i] * np.mean(myutils.find_k_largest(self.skills[team, i], 2))
        if not self.RESCALE or (not self.boundaries or 'score1' not in self.boundaries):
            return s1
        empirical_min = self.boundaries['score1']['min']
        empirical_max = self.boundaries['score1']['max']
        rescaled_s1 = (s1 - empirical_min) / (empirical_max - empirical_min)
        return rescaled_s1

    # score2 is defined as average value of score2 on each individual member
    # it returns a number in [0,1]
    def score2(self, teams):
        if self.t != len(teams):
            raise ValueError('The number of teams is different than problem\'s parameter initialization %d != %d' % (self.t, len(teams)))
        s2 = 0
        for team in teams:
            s2 += self.score2_single_team(team)
        s2 /= self.t
        return s2

    # Let std(t) be the standard deviation of the values in network connectivity among team members
    #  For team t, score2_single_team(t) = (1−2∗std(t))
    # it returns a number in [0,1]
    def score2_single_team(self, team):
        distances = []
        for i in range(len(team)):
            for j in range(i+1, len(team)):
                distances.append(self.network_connectivity[team[i], team[j]])
        s2 = 1 - 2 * np.std(distances)
        if not self.RESCALE or (not self.boundaries or 'score2' not in self.boundaries):
            return s2
        empirical_min = self.boundaries['score2']['min']
        empirical_max = self.boundaries['score2']['max']
        rescaled_s2 = (s2 - empirical_min) / (empirical_max - empirical_min)
        return rescaled_s2


    # def score3(self, teams):
    #     t = self.t
    #     m = self.m
    #     if t != len(teams):
    #         raise ValueError('The number of teams is different than problem\'s '
    #                          'parameter initialization %d != %d' % (t, len(teams)))
    #     optimal_bins = np.array([i / (t - 1) for i in range(t)])
    #     optimal_volumes = np.ones(t) / t
    #     distance_matrix = np.zeros((t, t))
    #     individuals_risks = np.sort(self.risk_takings[ np.ndarray.flatten(np.array(teams)) ])   # risk taking value of all individuals in teams
    #     if len(individuals_risks) != t * m:
    #         raise ValueError('The number of individuals in teams are different than problem\'s '
    #                          'parameter initialization %d != %d' % (t * m, len(individuals_risks)))
    #     # volumes: the amount of samples on each value
    #     # bins: a sorted bin values
    #     teams_volumes, teams_bins = np.histogram(individuals_risks, bins=t)
    #     teams_volumes = teams_volumes.astype(float, copy=False)   # to set it to be float
    #     # normalizing
    #     teams_volumes /= (m * t)
    #     teams_bin_centers = [(teams_bins[i]+teams_bins[i+1])/2 for i in range(t)]
    #     for i in range(t):
    #         for j in range(t):
    #             distance_matrix[i][j] = abs(optimal_bins[i] - teams_bin_centers[j])
    #     s3 = 1 - pyemd.emd(optimal_volumes, teams_volumes, distance_matrix)
    #     s3_best = self.score3_best(teams)
    #     s3_e = self.score3_e(teams)
    #     print('s3:', s3, 's3_e:', s3_e, 's3_best:', s3_best, '\n\n')
    #     if RESCALE:
    #         s3 = 13.56 * (s3 - 0.92)
    #     return s3

    # def score3_e(self, teams):    # emd without histogram works with average of team
    #     t = self.t
    #     if t != len(teams):
    #         raise ValueError(
    #             'The number of teams is different than problem\'s parameter initialization %d != %d' % (
    #             t, len(teams)))
    #     optimal_risk = np.array([i / (t - 1) for i in range(t)])
    #     team_risks = sorted(np.array([np.mean(self.risk_takings[teams[i]]) for i in range(t)]))  # average of members'
    #     return 2 - myutils.compute_emd(optimal_risk, team_risks)


    # ---== Risk-taking ==---
    # Define the ideal risk-taking distribution across t teams as t \
    #   masses of 1/t located at positions 1/2t,3/2t,···,2t−1/2t
    #  Call this distribution RI. Given a set T of t teams, let R(T) be the normalized distribution of
    #   the risk-taking behavior of the t×m individuals
    #    For teams T, score3_emd(T) = 1−EMD(RI,R(T)).
    # It returns a number in [0,1]
    def score3(self, teams, number_of_teams=None):  # emd without histogram works with each individual person
        if number_of_teams is None:
            t = self.t
        else:
            t = number_of_teams
        m = self.m
        team_risks = np.array([np.mean(self.risk_takings[teams[i]]) for i in range(t)])
        ordered_teams = [x for _, x in sorted(zip(team_risks, teams))]  # in ascending order for risks
        d = 0
        for i, team in enumerate(ordered_teams):
            if t != 1:
                optimal_team_risk = i / (t-1)
            else:
                optimal_team_risk = 0.5
            d += myutils.compute_emd(np.ones(m)*optimal_team_risk, self.risk_takings[team])
        s3 = 1 - d/t
        if not self.RESCALE or (not self.boundaries or 'score3' not in self.boundaries):
            return s3
        empirical_min = self.boundaries['score3']['min']
        empirical_max = self.boundaries['score3']['max']
        rescaled_s3 = (s3 - empirical_min) / (empirical_max - empirical_min)
        return rescaled_s3


    """It validates if the teams parameter is valid w.r.t. problem info or not"""
    ###Currently not using inside the problem (only in main)###
    def validate_solution(self, teams):
        if len(teams) != self.t:
            return False
        for team in teams:
            if (len(team) != self.m) or (len(np.unique(team)) != self.m):
                return False
        members = np.ndarray.flatten(np.array(teams))
        if len(np.unique(members)) != self.t * self.m:
            return False
        if np.where(members > self.n)[0] or np.where(members < 0)[0]:
            return False
        return True


    # # it DOESN'T returns a number in [0,1]
    # def score3_meanstd(self, teams):
    #     t = self.t
    #     eps = 0.001
    #     means = np.zeros(t)
    #     stds = np.zeros(t)
    #     for i in range(t):
    #         risks_i = self.risk_takings[teams[i]]  # risks taking values for team i
    #         means[i] = np.mean(risks_i)
    #         stds[i] = np.std(risks_i)
    #     s3 = 0
    #     for i in range(t):
    #         s3_team = 0
    #         for j in range(t):
    #             s3_team += np.power((means[i] - means[j]), 2)
    #         s3_team /= (t * (np.power(stds[i], 2) + eps))
    #         s3 += s3_team
    #     s3 /= t
    #     return s3

