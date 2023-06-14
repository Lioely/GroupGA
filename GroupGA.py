import numpy
import random as ra
import math
import numpy as np
import time


def division_two(num):
    tmp = num
    res = []
    while tmp >= 2:
        res.append(tmp % 2)
        tmp = tmp // 2
    res.append(tmp)
    res.reverse()
    return res


# 计算0-1编码的最大长度函数
def cal_max_num(num):
    for i in range(0, 100):
        if 2**i >= num:
            return i


def find_best(fitness):
    return np.argmax(fitness)


def dec2bin(x):
    x -= int(x)
    bins = []
    while x and len(bins) < 10:
        x *= 2
        bins.append(1 if x >= 1. else 0)
        x -= int(x)
    while len(bins) < 10:
        bins.append(0)
    return bins


def bin2dec(b):
    d = 0
    for i, x in enumerate(b):
        d += 2**(-i - 1) * x
    return d


def reverse_species(species):
    return np.where(species, 0, 1)


class GA:
    def __init__(self, function, species_num, individual_dim, max_x, min_x,
                 iteration):
        """
        为了简化实验，默认是0-1编码，默认是自适应的参数选择，默认是头-尾选择
        :param function:适应值函数
        :param species_num:种群数
        :param iteration:迭代次数
        :param individual_dim:个体的维度，是一个列表
        :param max_x:每个维度的最大值
        :param min_x:每个维度的最小值
        """

        self.function = function
        self.groupA_mutations = [ra.uniform(0, 1) for _ in range(species_num)]
        self.groupB_mutations = [ra.uniform(0, 1) for _ in range(species_num)]
        self.groupC_mutations = [ra.uniform(0, 1) for _ in range(species_num)]

        self.groupA_crossover = [ra.uniform(0, 1) for _ in range(species_num)]
        self.groupB_crossover = [ra.uniform(0, 1) for _ in range(species_num)]
        self.groupC_crossover = [ra.uniform(0, 1) for _ in range(species_num)]

        self.species_num = species_num
        self.iteration = iteration

        self.individual_dim = individual_dim

        # 记录每个维度的编码长度
        self.length_dim = []

        # 记录fitness result
        self.res = []
        self.best_res = []

        # 记录time cost
        self.time = []

        if not isinstance(max_x, list):
            self.max_x = [max_x]
        else:
            self.max_x = max_x

        if not isinstance(min_x, list):
            self.min_x = [min_x]
        else:
            self.min_x = min_x

        if individual_dim != len(max_x) and individual_dim != len(min_x):
            raise "The dimension and range of x must have the same shape"

        self.species = None
        self.species_float = None

    def cal_zero_one(self, dim, code, code_float):
        sum_ = 0
        code = list(code)
        code.reverse()
        for i, j in enumerate(code):
            if j:
                sum_ += 2**i
        sum_ += bin2dec(code_float)
        return sum_ + self.min_x[dim]

    # 计算0-1编码的最大长度函数
    def cal_long(self):
        for i, j in zip(self.max_x, self.min_x):
            length = cal_max_num(i - j + 1)
            self.length_dim.append(length)

    def decoder(self, species, species_float):
        values = []
        for individual, individual_float in zip(species, species_float):
            tmp = []
            for dim in range(self.individual_dim):
                every_dim = individual[dim]
                float_every_dim = individual_float[dim]
                value = self.cal_zero_one(dim, every_dim, float_every_dim)
                if value > self.max_x[dim] or value < self.min_x[dim]:
                    value = ra.uniform(self.min_x[dim], self.max_x[dim])
                tmp.append(value)
            values.append(np.array(tmp))
        values = np.array(values)
        return values

    def encoder(self, values):
        species = []
        species_float = []
        for value in values:
            individual = []
            individual_float = []
            for dim in range(self.individual_dim):
                value_dim = value[dim]
                value_float, value_int = math.modf(value_dim)
                code = division_two(value_int)
                code_float = dec2bin(round(value_float, 3))
                while len(code) < self.length_dim[dim]:
                    code.insert(0, 0)
                individual_float.append(np.array(code_float))
                individual.append(np.array(code))
            species.append(np.array(individual))
            species_float.append(np.array(individual_float))
        return np.array(species), np.array(species_float)

    def initialize(self):
        values = None
        for dim in range(self.individual_dim):
            mi_x = self.min_x[dim]
            ma_x = self.max_x[dim]
            ra.seed()
            tmp = np.random.uniform(0, ma_x - mi_x + 1, (self.species_num, 1))
            if values is None:
                values = tmp
            else:
                values = np.concatenate([values, tmp], axis=1)
        species, species_float = self.encoder(values)  # [[1,1,1],[2,2,2]]
        self.species = species
        self.species_float = species_float

    def adaptive_mutation_rate(self, group, fitness):
        if group == "A":
            mutations = self.groupA_mutations
        elif group == "B":
            mutations = self.groupB_mutations
        else:
            mutations = self.groupC_mutations

        max_fit = np.max(fitness)
        mean_fit = np.mean(fitness)
        for i, fit in enumerate(fitness):
            if fit != mean_fit:
                mutations[i] *= round((max_fit - fit) / (max_fit - mean_fit),
                                      2)
        return np.array(mutations)

    def adaptive_crossover_rate(self, group, fitness):
        if group == "A":
            crossover = self.groupA_crossover
        elif group == "B":
            crossover = self.groupB_crossover
        else:
            crossover = self.groupC_crossover

        max_fit = np.max(fitness)
        mean_fit = np.mean(fitness)
        for i, fit in enumerate(fitness):
            if fit != mean_fit:
                crossover[i] *= round((max_fit - fit) / (max_fit - mean_fit),
                                      2)
        return np.array(crossover)

    def selection(self, groupA, groupB, groupC, groupA_float, groupB_float,
                  groupC_float, fitnessA, fitnessB, fitnessC):
        mean_A = abs(np.mean(fitnessA))
        mean_B = abs(np.mean(fitnessB))

        sum_ = mean_A + mean_B
        raitoA = round(mean_A / sum_, 1)
        raitoB = round(mean_B / sum_, 1)

        numC = self.species_num // 2
        numA = int((self.species_num - numC) * raitoA)
        numB = self.species_num - numC - numA  # int(self.species_num * raitoB)

        # get C
        top_num_C = numC // 2
        ind = np.argpartition(fitnessC, -top_num_C)[-top_num_C:]
        top_c = groupC[ind]
        top_c_re = reverse_species(top_c)
        top_c = np.concatenate([top_c, top_c_re], axis=0)

        top_c_float = groupC_float[ind]
        top_c_re_float = reverse_species(top_c_float)
        top_c_float = np.concatenate([top_c_float, top_c_re_float], axis=0)

        mutations_c = self.groupC_mutations[ind]
        mutations_c = np.concatenate([mutations_c, mutations_c], axis=0)
        crossover_c = self.groupC_crossover[ind]
        crossover_c = np.concatenate([crossover_c, crossover_c], axis=0)

        # getA,getB
        top_numA = numA // 2
        low_numA = numA - top_numA

        top_numB = numB // 2
        low_numB = numB - top_numB

        ind = np.argpartition(fitnessA, -top_numA)[-top_numA:]
        top_a = groupA[ind]
        top_a_float = groupA_float[ind]
        top_mutations_a = self.groupA_mutations[ind]
        top_crossover_a = self.groupA_crossover[ind]

        ind = np.argpartition(fitnessB, -top_numB)[-top_numB:]
        top_b = groupB[ind]
        top_b_float = groupB_float[ind]
        top_mutations_b = self.groupB_mutations[ind]
        top_crossover_b = self.groupB_crossover[ind]

        ind = np.argpartition(fitnessA, low_numA)[:low_numA]
        low_a = groupA[ind]
        low_a_float = groupA_float[ind]
        low_mutations_a = self.groupA_mutations[ind]
        low_crossover_a = self.groupA_crossover[ind]

        ind = np.argpartition(fitnessB, low_numB)[:low_numB]
        low_b = groupB[ind]
        low_b_float = groupB_float[ind]
        low_mutations_b = self.groupB_mutations[ind]
        low_crossover_b = self.groupB_crossover[ind]

        species = np.concatenate((top_c, top_a, top_b, low_a, low_b))
        species_float = np.concatenate(
            (top_c_float, top_a_float, top_b_float, low_a_float, low_b_float))
        mutations = np.concatenate(
            (mutations_c, top_mutations_a, top_mutations_b, low_mutations_a,
             low_mutations_b))
        crossover = np.concatenate(
            (crossover_c, top_crossover_a, top_crossover_b, low_crossover_a,
             low_crossover_b))

        self.species = species
        self.species_float = species_float
        self.groupA_mutations = mutations
        self.groupB_mutations = mutations
        self.groupC_mutations = mutations

        self.groupA_crossover = crossover
        self.groupB_crossover = crossover
        self.groupC_crossover = crossover

    def crossover(self, group, species, species_float):
        if group == "A":
            # 均匀交叉
            species_float, species, crs_pro = self.UniformCrossover(
                species, species_float, self.groupA_crossover)
        else:
            # 模拟二进制交叉
            species_float, species, crs_pro = self.SimulatedBinaryCrossover(
                species, species_float, self.groupB_crossover)

        return species_float, species, crs_pro

    def mutation(self, group, species, current_generation):
        if group == "A":
            # 均匀变异
            species = self.UniformMutation(species, self.groupA_mutations)
        else:
            # 非一致变异
            species = self.NonUniformMutation(species, self.groupB_mutations,
                                              current_generation,
                                              self.iteration)

        return species

    # 均匀交叉
    def UniformCrossover(self, species, species_float, crossover):
        individual_dim = self.individual_dim
        new_species = []
        new_species_float = []
        new_crossover = []
        while len(new_species) < self.species_num:
            ra.seed()
            ind1 = ra.randint(0, self.species_num // 2 - 1)
            ind2 = ra.randint(self.species_num // 2, self.species_num - 1)
            individual1 = species[ind1]
            individual2 = species[ind2]
            individual1_float = species_float[ind1]
            individual2_float = species_float[ind2]
            for dim in range(individual_dim):
                ra.seed()
                pro1 = ra.random()
                pro2 = ra.random()
                gen1 = individual1[dim]
                gen2 = individual2[dim]

                gen1_float = individual1_float[dim]
                gen2_float = individual2_float[dim]
                if pro1 < crossover[ind1]:
                    individual1[dim] = gen2
                    individual1_float[dim] = gen2_float
                if pro2 < crossover[ind2]:
                    individual2[dim] = gen1
                    individual2_float[dim] = gen1_float
            new_crossover.append(np.array(crossover[ind1]))
            new_crossover.append(np.array(crossover[ind2]))
            new_species.append(np.array(individual1))
            new_species.append(np.array(individual2))
            new_species_float.append(np.array(individual1_float))
            new_species_float.append(np.array(individual2_float))

        if len(new_species) > self.species_num:
            new_species.pop()
            new_species_float.pop()

        if len(new_crossover) > self.species_num:
            new_crossover.pop()

        crossover = np.array(new_crossover)
        species = np.array(new_species)
        species_float = np.array(new_species_float)
        return species_float, species, crossover

    # 模拟二进制交叉
    def SimulatedBinaryCrossover(self, species, species_float, crossover):
        individual_dim = self.individual_dim
        new_species = []
        new_species_float = []
        new_crossover = []
        while len(new_species) < self.species_num:
            ra.seed()
            ind1 = ra.randint(0, self.species_num // 2 - 1)
            ind2 = ra.randint(self.species_num // 2, self.species_num - 1)
            individual1 = species[ind1]
            individual2 = species[ind2]
            individual1_float = species_float[ind1]
            individual2_float = species_float[ind2]
            pro = ra.random()
            if pro < crossover[ind1] or pro < crossover[ind2]:
                for dim in range(individual_dim):
                    ra.seed()
                    r = ra.random()
                    if r >= 0.5:
                        b = (2 * r)**(1 / (0.1 + 1))
                    else:
                        b = (1 / (2 * (1 - r)))**(1 / (0.1 + 1))
                    p1 = individual1[dim]
                    p2 = individual2[dim]
                    p1_float = individual1_float[dim]
                    p2_float = individual2_float[dim]

                    individual1[dim] = 0.5 * ((1 + b) * p1 + (1 - b) * p2)
                    individual2[dim] = 0.5 * ((1 - b) * p1 + (1 + b) * p2)
                    individual1_float[dim] = 0.5 * ((1 + b) * p1_float +
                                                    (1 - b) * p2_float)
                    individual2_float[dim] = 0.5 * ((1 - b) * p1_float +
                                                    (1 + b) * p2_float)

            new_crossover.append(np.array(crossover[ind1]))
            new_crossover.append(np.array(crossover[ind2]))
            new_species.append(np.array(individual1))
            new_species.append(np.array(individual2))
            new_species_float.append(np.array(individual1_float))
            new_species_float.append(np.array(individual2_float))

        if len(new_species) > self.species_num:
            new_species.pop()
            new_species_float.pop()

        if len(new_crossover) > self.species_num:
            new_crossover.pop()

        crossover = np.array(new_crossover)
        species = np.array(new_species)
        species_float = np.array(new_species_float)
        return species_float, species, crossover

    # 均匀变异
    def UniformMutation(self, species, mutations):
        individual_dim = self.individual_dim
        new_species = []
        for individual, mu in zip(species, mutations):
            new_individual = individual
            for dim in range(individual_dim):
                ra.seed()
                pro = ra.random()
                if pro < mu:
                    new_individual[dim] = abs(new_individual[dim] - 1)

            new_species.append(np.array(new_individual))

        species = np.array(new_species)
        return species

    # 非一致变异
    def NonUniformMutation(self, species, mutations, current_generation,
                           max_generations):
        individual_dim = self.individual_dim
        new_species = []
        pro = 1 * math.exp(-current_generation / max_generations)
        for individual, mu in zip(species, mutations):
            new_individual = individual
            for dim in range(individual_dim):
                if pro < mu:
                    new_individual[dim] = abs(new_individual[dim] - 1)
            new_species.append(np.array(new_individual))
        species = np.array(new_species)
        return species

    def fit(self):
        # 初始化种群，交叉变异概率
        self.cal_long()
        self.initialize()
        fitness = self.function(self.decoder(self.species, self.species_float))
        self.groupA_mutations = self.adaptive_mutation_rate("A", fitness)
        self.groupB_mutations = self.adaptive_mutation_rate("B", fitness)
        self.groupC_mutations = self.adaptive_mutation_rate("C", fitness)

        self.groupA_crossover = self.adaptive_crossover_rate("A", fitness)
        self.groupB_crossover = self.adaptive_crossover_rate("B", fitness)
        self.groupC_crossover = self.adaptive_crossover_rate("C", fitness)

        mu_time = 0
        cro_time = 0
        fit_time = 0
        sel_time = 0
        all_time = 0
        # 迭代
        all_start = time.time()
        for iters in range(1, self.iteration):
            # split three groups
            groupA = self.species
            groupB = self.species
            groupC = self.species
            groupA_float = self.species_float
            groupB_float = self.species_float
            groupC_float = self.species_float

            # mutations
            start = time.time()
            groupA = self.mutation("A", groupA, iters)
            groupB = self.mutation("B", groupB, iters)
            groupA_float = self.mutation("A", groupA_float, iters)
            groupB_float = self.mutation("B", groupB_float, iters)
            end = time.time()
            mu_time += end - start

            # crossover
            start = time.time()
            groupA_float, groupA, crs_A = self.crossover(
                "A", groupA, groupA_float)
            groupB_float, groupB, crs_B = self.crossover(
                "B", groupB, groupB_float)
            end = time.time()
            cro_time += end - start

            self.groupA_crossover = crs_A
            self.groupB_crossover = crs_B

            # cal fitness
            start = time.time()
            valueA, valueB, valueC = self.decoder(
                groupA, groupA_float), self.decoder(
                    groupB, groupB_float), self.decoder(groupC, groupC_float)
            fitnessA, fitnessB, fienessC = self.function(
                valueA), self.function(valueB), self.function(valueC)
            end = time.time()
            fit_time += end - start

            # adjust & selection
            start = time.time()
            groupA, groupA_float = self.encoder(valueA - np.array(self.min_x))
            groupB, groupB_float = self.encoder(valueB - np.array(self.min_x))
            groupC, groupC_float = self.encoder(valueC - np.array(self.min_x))
            self.selection(groupA, groupB, groupC, groupA_float, groupB_float,
                           groupC_float, fitnessA, fitnessB, fienessC)
            end = time.time()
            sel_time += end - start

            # find_best
            species_fitness = self.function(
                self.decoder(self.species, self.species_float))
            ind = find_best(species_fitness)
            print(f"----------iteration:{iters}----------")
            print(f"best fitness: {species_fitness[ind]}")
            print(f"----------iteration:{iters}----------")
            self.res.append(species_fitness)
            self.best_res.append(species_fitness[ind])
        all_end = time.time()
        all_time += all_end - all_start
        self.time.extend([all_time, mu_time, cro_time, fit_time, sel_time])


"""
example:

def function1(values):
    # x_i ** 2 - x_i + 1
    fitness = []
    for value in values:
        res = 0
        for i in value:
            res += (i - 1)**2
        fitness.append(res)
    return np.array(fitness)
ga = GA(function1, 1000, 10, [100 for _ in range(10)],
        [-100 for _ in range(10)], 100)
ga.fit()
print(ga.time)
"""
