import numpy as np
import random as ra
import math


def find_best(fitness):
    return np.argmax(fitness)


class SaDe:
    def __init__(self, function, NP, LP, individual_dim, max_x, min_x,
                 iteration):
        """
        根据论文
        :param function:适应值函数
        :param NP:是种群个数
        :param LP: forward learning step
        :param individual_dim: 是种群的特征维度
        :param max_x: 每个维度的最大值，list-> numpy
        :param min_x: 每个维度的最小值，list-> numpy
        :param iteration : 迭代次数
        """
        self.function = function
        # F是每个个体的交叉变异中的scale因子，numpy :[NP,]
        self.F = np.array([
            np.array([0.3 * ra.random() + 0.5 for _ in range(4)])
            for _ in range(NP)
        ])
        # CR是每个个体的交叉概率, numpy: [NP,4]
        self.CR = np.array([
            np.array([0.1 * ra.random() + 0.5 for _ in range(4)])
            for _ in range(NP)
        ])
        # K 策略池中策略个数，4
        self.K = 4
        # LP forward learning step
        self.LP = LP
        # NP
        self.NP = NP
        #
        self.iterations = iteration

        self.best_fitness = []

        if not isinstance(max_x, list):
            self.max_x = np.array([max_x])
        else:
            self.max_x = np.array(max_x)

        if not isinstance(min_x, list):
            self.min_x = np.array([min_x])
        else:
            self.min_x = np.array(min_x)

        if individual_dim != len(max_x) and individual_dim != len(min_x):
            raise "The dimension and range of x must have the same shape"

        self.individual_dim = individual_dim

        self.species = None
        self.best = None
        self.best_fit = None

        self.p_k = [0 for _ in range(4)]
        self.s_field, self.f_field = np.array([0, 0, 0,
                                               0]), np.array([0, 0, 0, 0])
        self.all_s_field = []
        self.all_f_field = []

    def adjust(self, individual):
        for i in range(self.individual_dim):
            if individual[i] > self.max_x[i] or individual[i] < self.min_x[i]:
                individual[i] = round(ra.uniform(self.min_x[i], self.max_x[i]),
                                      3)
        return individual

    def DE_1(self, individual, cr, F):
        ra.seed()
        pro = ra.random()
        if pro < cr:
            ra.seed()
            index = ra.sample(range(0, self.NP - 1), 3)
            index1, index2, index3 = index[0], index[1], index[2]
            new_individual = self.species[index1] + F * (self.species[index2] -
                                                         self.species[index3])
        else:
            new_individual = individual
        return new_individual

    def DE_2(self, individual, cr, F):
        ra.seed()
        pro = ra.random()
        if pro < cr:
            ra.seed()
            index = ra.sample(range(0, self.NP - 1), 4)
            index1, index2, index3, index4 = index[0], index[1], index[
                2], index[3]
            new_individual = individual + F * (self.best - individual) + F * (
                self.species[index1] - self.species[index2]) + F * (
                    self.species[index3] - self.species[index4])
        else:
            new_individual = individual
        return new_individual

    def DE_3(self, individual, cr, F):
        ra.seed()
        pro = ra.random()
        if pro < cr:
            ra.seed()
            index = ra.sample(range(0, self.NP - 1), 5)
            index1, index2, index3, index4, index5 = index[0], index[1], index[
                2], index[3], index[4]
            new_individual = self.species[index1] + F * (
                self.species[index2] - self.species[index3]) + F * (
                    self.species[index4] - self.species[index5])
        else:
            new_individual = individual
        return new_individual

    def DE_4(self, individual, cr, F):
        ra.seed()
        pro = ra.random()
        if pro < cr:
            ra.seed()
            index = ra.sample(range(0, self.NP - 1), 3)
            index1, index2, index3 = index[0], index[1], index[2]
            new_individual = individual + ra.random() * (
                self.species[index1] -
                individual) + F * (self.species[index2] - self.species[index3])
        else:
            new_individual = individual
        return new_individual

    def initialize(self):
        # initialize species , best_individual , q_k , s_field , f_field
        values = []
        for _ in range(self.NP):
            value = []
            for dim in range(self.individual_dim):
                mi_x = self.min_x[dim]
                ma_x = self.max_x[dim]
                ra.seed()
                value.append(ra.uniform(mi_x, ma_x))
            values.append(np.array(value))
        species = np.array(values)
        self.species = species
        fitness = self.function(self.species)
        best_ind = find_best(fitness)
        self.best = np.array(self.species[best_ind])
        self.best_fit = self.function([self.species[best_ind]])

        new_species = []
        # LP initialize  p_k ,s_filed,f_filed
        for cnt, individual in enumerate(self.species):
            sub_fitness = []
            new_individual = []
            F = 0.3 * ra.random() + 0.5
            new_individual1 = self.adjust(
                np.array(self.DE_1(individual, self.CR[cnt][0], F)))
            sub_fitness.append(self.function([new_individual1]))
            new_individual.append(new_individual1)

            new_individual2 = self.adjust(
                np.array(self.DE_2(individual, self.CR[cnt][1], F)))
            sub_fitness.append(self.function([new_individual2]))
            new_individual.append(new_individual2)

            new_individual3 = self.adjust(
                np.array(self.DE_3(individual, self.CR[cnt][2], F)))
            sub_fitness.append(self.function([new_individual3]))
            new_individual.append(new_individual3)

            new_individual4 = self.adjust(
                np.array(self.DE_4(individual, self.CR[cnt][3], F)))
            sub_fitness.append(self.function([new_individual4]))
            new_individual.append(new_individual4)

            ind = np.argmax(np.array(sub_fitness))
            self.s_field[ind] += 1
            self.f_field += 1
            self.f_field[ind] -= 1
            new_species.append(new_individual[ind])
        # 大S
        self.all_s_field.append(self.s_field)
        self.all_f_field.append(self.f_field)
        S_field = np.array([
            self.s_field[i] / (np.sum(self.s_field) + np.sum(self.f_field))
            for i in range(4)
        ])
        # p_k
        self.p_k = np.array([S_field[i] / np.sum(S_field) for i in range(4)])
        self.species = np.array(new_species)

    def fit(self):
        self.initialize()
        cnt = 0
        All_DE = [self.DE_1, self.DE_2, self.DE_3, self.DE_4]
        for G in range(1, self.iterations + 1):
            new_species = []
            pre_s = self.s_field
            pre_f = self.f_field
            sum_qk = np.cumsum(self.p_k)
            pro = ra.random()
            for _ in range(self.NP):
                for i in range(len(sum_qk)):
                    if pro < sum_qk[i]:
                        self.s_field[i] += 1
                        self.f_field += 1
                        self.f_field[i] -= 1
                        break
            self.all_s_field.append(self.s_field - pre_s)
            self.all_f_field.append(self.f_field - pre_f)
            if G > self.LP:
                cnt += 1
            # 大S
            S_field = np.array([
                self.s_field[i] /
                (np.sum(np.array(self.all_s_field)[cnt:, i]) +
                 np.sum(np.array(self.all_f_field)[cnt:, i])) for i in range(4)
            ])
            # p_k
            self.p_k = np.array(
                [S_field[i] / np.sum(S_field) for i in range(4)])
            cum_p = self.p_k.cumsum()
            for i, individual in enumerate(self.species):
                # strategy pro
                de_method = All_DE[3]
                ra.seed()
                pro = ra.random()
                # select K
                cr = 0.8
                for j in range(4):
                    if pro >= cum_p[j]:
                        de_method = All_DE[j]
                        cr_median = np.median(self.CR[:][j])
                        cr = 0.1 * ra.random() + cr_median + 1
                        self.CR[i][j] = cr
                        break
                ra.seed()
                F = 0.3 * ra.random() + 0.8
                new_individual = de_method(individual, cr, F)
                new_individual - self.adjust(new_individual)
                new_fitness = self.function([new_individual])
                fitness = self.function([individual])
                if new_fitness > fitness:
                    new_species.append(new_individual)
                else:
                    new_species.append(individual)
                if new_fitness > self.best_fit:
                    self.best = new_individual
                    self.best_fit = new_fitness

            self.species = np.array(new_species)
            species_fitness = self.function(self.species)
            ind = find_best(species_fitness)
            print(f"----------iteration:{G}----------")
            print(f", best fitness: {species_fitness[ind]}")
            print(f"----------iteration:{G}----------")
            self.best_fitness.append(species_fitness[ind])


#SaDE
"""
example
import time
def function1(values):
    # x_i ** 2 - x_i + 1
    fitness = []
    for value in values:
        res = 0
        for i in value:
            res += (i - 50)**2 - 10 * math.cos(2 * math.pi * (i - 50)) + 10
        fitness.append(res)
    return np.array(fitness)

sade = SaDe(function1, 100, 50, 10, [100 for _ in range(10)], [-100 for _ in range(10)], 1000)
start = time.time()
sade.fit()# self.best_fitness
end = time.time()
print(f"inference time {end-start}")
"""
