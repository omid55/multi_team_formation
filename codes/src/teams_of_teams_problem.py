# Omid55
# teams of teams problem data generation
# it includes the generation part plus the objective function

# imports
import numpy as np
import pyemd

RESCALE = True

# n: number of individuals
# m: number of team members
# k: number of teams
# s: number of skills
class Problem:
    def __init__(self, n, s, m, k, alpha=1/3, beta=1/3):
        self.n = n
        self.m = m
        self.k = k
        self.s = s
        self.alpha = alpha
        self.beta = beta
        self._generate()
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
        # np.random.seed(1)  # to make all random values same   << COMMENT IT FOR ACTUAL RUN >>
        # skills of every person: every person has a list of s elements describing his/her skill values fall in [0-1].
        self.skills = np.random.rand(self.n, self.s)
        # risk taking of every person: it is also a value between [0,1].
        self.risk_takings = np.random.rand(self.n)
        # network connectivity (or distance) is a matrix of n*n: we always can for any
        #   graph, compute the shortest path among all nodes
        #   and normalize it between [0,1]. We assume, relationships are not directed.
        C = np.random.rand(self.n, self.n)
        C = np.maximum(C, C.T)
        for i in range(self.n):
            C[i, i] = 0
        self.network_connectivity = C

    # teams: is a list of teams which each team is a list of indices (individual indices)
    #   for instance: teams = [[1,2,4],[0,3,5]]
    #   n: #individuals
    #   k: #teams
    #   m: size of each team
    def objective_function(self, teams):
        # score based on 3 different factors
        score = (1 - self.alpha - self.beta) * self.score1(teams) + self.alpha * self.score2(
            teams) + self.beta * self.score3(teams)
            # teams) + self.beta * self.score3_meanstd(teams)
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
            s1 += np.mean(sorted(self.skills[team,i])[-2:])
        s1 /= self.s
        if RESCALE:
            s1 = 7 * (s1 - 0.79)
        return s1

    # score2 is defined as average value of score2 on each individual member
    # it returns a number in [0,1]
    def score2(self, teams):
        if self.k != len(teams):
            raise ValueError('The number of teams is different than problem\'s parameter initialization %d != %d' % (self.k, len(teams)))
        s2 = 0
        for team in teams:
            s2 += self.score2_single_team(team)
        s2 /= self.k
        return s2

    # Let std(t) be the standard deviation of the values in network connectivity among team members
    #  For team t, score2_single_team(t) = (1−2∗std(t))
    # it returns a number in [0,1]
    def score2_single_team(self, team):
        distances = []
        for i in range(len(team)):
            for j in range(i+1, len(team)):
                distances.append(self.network_connectivity[team[i]][team[j]])
        s2 = 1 - 2 * np.std(distances)
        if RESCALE:
            s2 = 13.8 * (s2 - 0.5)
        return s2

    # Define the ideal risk-taking distribution across k teams as k \
    #   masses of 1/k located at positions 1/2k,3/2k,···,2k−1/2k
    #  Call this distribution RI. Given a set T of k teams, let R(T) be the normalized distribution of
    #   the risk-taking behavior of the k×m individuals
    #    For teams T, score3_emd(T) = 1−EMD(RI,R(T)).
    # it returns a number in [0,1]
    #   << CHECK HERE >> IT DOESN'T INCLUDE STD INSIDE TEAM
    def score3(self, teams):
        k = self.k
        m = self.m
        if k != len(teams):
            raise ValueError('The number of teams is different than problem\'s parameter initialization %d != %d' % (k, len(teams)))
        optimal_bins = np.array([i / (k - 1) for i in range(k)])
        optimal_volumes = m * np.ones(k)
        distance_matrix = np.zeros((k, k))
        individuals_risks = np.sort(self.risk_takings[ np.ndarray.flatten(np.array(teams)) ])   # risk taking value of all individuals in teams
        if len(individuals_risks) != k * m:
            raise ValueError('The number of individuals in teams are different than problem\'s parameter initialization %d != %d' % (k * m, len(individuals_risks)))
        # volumes: the amount of samples on each value
        # bins: a sorted bin values
        teams_volumes, teams_bins = np.histogram(individuals_risks, bins=k)
        teams_volumes = teams_volumes.astype(float, copy=False)   # to set it to be float
        # normalizing
        optimal_volumes /= (m * k)
        teams_volumes /= (m * k)
        teams_bin_centers = [(teams_bins[i]+teams_bins[i+1])/2 for i in range(k)]
        for i in range(k):
            for j in range(k):
                distance_matrix[i][j] = abs(optimal_bins[i] - teams_bin_centers[j])
        s3 = 1 - pyemd.emd(optimal_volumes, teams_volumes, distance_matrix)
        if RESCALE:
            s3 = 13.56 * (s3 - 0.92)
        return s3

    # def score3_avgteam(self, teams):    # JUST FOR BACK UP
    #     if self.k != len(teams):
    #         raise ValueError(
    #             'The number of teams is different than problem\'s parameter initialization %d != %d' % (
    #             self.k, len(teams)))
    #     optimal_risk = np.array([i / (self.k - 1) for i in range(self.k)])
    #     team_risks = sorted(np.array(
    #         [np.mean(self.risk_takings[teams[i]]) for i in range(self.k)]))  # average of members' risks
    #     optimal_volumes = self.m * np.ones(self.k)
    #     distance_matrix = np.zeros((self.k, self.k))
    #     for i in range(self.k):
    #         for j in range(self.k):
    #             distance_matrix[i][j] = abs(team_risks[i] - optimal_risk[j])
    #     return 1 - pyemd.emd(np.ones(self.k), optimal_volumes, distance_matrix)

    # it DOESN'T returns a number in [0,1]
    def score3_meanstd(self, teams):
        k = self.k
        eps = 0.001
        means = np.zeros(k)
        stds = np.zeros(k)
        for i in range(k):
            risks_i = self.risk_takings[teams[i]]  # risks taking values for team i
            means[i] = np.mean(risks_i)
            stds[i] = np.std(risks_i)
        s3 = 0
        for i in range(k):
            s3_team = 0
            for j in range(k):
                s3_team += np.power((means[i] - means[j]), 2)
            s3_team /= (k * (np.power(stds[i], 2) + eps))
            s3 += s3_team
        s3 /= k
        if RESCALE:
            s3 = 13.56 * (s3 - 0.92)
        return s3


