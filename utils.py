from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_random_state(seed):
    return np.random.RandomState(seed)


def random_boolean_1D_array(length, random_state):
    return random_state.choice([True, False], length)


def bit_flip(bit_string, random_state):
    neighbour = bit_string.copy()
    index = random_state.randint(0, len(neighbour))
    neighbour[index] = not neighbour[index]

    return neighbour


def parametrized_iterative_bit_flip(prob):
    def iterative_bit_flip(bit_string, random_state):
        neighbor = bit_string.copy()
        for index in range(len(neighbor)):
            if random_state.uniform() < prob:
                neighbor[index] = not neighbor[index]
        return neighbor

    return iterative_bit_flip


def random_float_1D_array(hypercube, random_state):
    return np.array([random_state.uniform(tuple_[0], tuple_[1])
                     for tuple_ in hypercube])


def random_float_cbound_1D_array(dimensions, l_cbound, u_cbound, random_state):
    return random_state.uniform(lower=l_cbound, upper=u_cbound, size=dimensions)


def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array([random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
    return ball_mutation


def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=len(point.shape) % 2 - 1)


def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r

def two_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    point2 = random_state.randint(len_)
    if point > point2:
        off1_r = np.concatenate((p1_r[0:point2], p2_r[point2:point],p1_r[point:len_]))
        off2_r = np.concatenate((p2_r[0:point2], p1_r[point2:point],p2_r[point:len_]))
    elif point <= point2:
        off1_r = np.concatenate((p1_r[0:point], p2_r[point:point2],p1_r[point2:len_]))
        off2_r = np.concatenate((p2_r[0:point], p1_r[point:point2],p2_r[point2:len_]))

    return off1_r, off2_r


def uniform_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    p_c  = 0.75
    off1_r = []
    off2_r = []
    for i in range(len_):
        if random_state.random_sample() >= p_c:
            off1_r.append(p2_r[i])
            off2_r.append(p1_r[i])
        else:
            off1_r.append(p2_r[i])
            off2_r.append(p1_r[i])

    return off1_r, off2_r

def inversion(p1_r, p2_r, random_state): # test it once you will go back to
    len_ = len(p1_r)
    p_c  = 0.75
    off1_r = p1_r
    off2_r = p2_r
    k = random_state.uniform() * 10
    loc = 640 + k*10
    inv = off1_r[loc:loc+10]
    inv = np.inv[-1]
    off1_r[loc:loc+10] = inv
    k = random_state.uniform() * 10
    loc = 640 + k*10
    inv = off2_r[loc:loc+10]
    inv = np.inv[-1]
    off2_r[loc:loc+10] = inv
    return off1_r, off2_r

def generate_cbound_hypervolume(dimensions, l_cbound, u_cbound):
  return [(l_cbound, u_cbound) for _ in range(dimensions)]


def parametrized_uniform_crossover(prob):
    def uniform_crossover(p1_r, p2_r, random_state):
        len_ = len(p1_r)
        p_c = 0.75
        off1_r = []
        off2_r = []
        for i in range(len_):
            if random_state.random_sample() >= p_c:
                off1_r.append(p2_r[i])
                off2_r.append(p1_r[i])
            else:
                off1_r.append(p2_r[i])
                off2_r.append(p1_r[i])

        return off1_r, off2_r
    return uniform_crossover


def geometric_crossover(p1_r, p2_r,p3_r, p4_r, random_state):
    len_ = len(p1_r)
    off_1 = []
    off_2 = []
    for i in range(len_):
        r = random_state.uniform()
        off_1.append(r*p1_r[i] + (1-r)*p2_r[i])
        r = random_state.uniform()
        off_2.append(r * p3_r[i] + (1 - r) * p4_r[i])
    return off_1, off_2

def geometric_crossover_ver2(p1_r, p2_r,p3_r, p4_r, random_state):
    len_ = len(p1_r)
    off_1 = []
    off_2 = []
    for i in range(len_):
        r = random_state.uniform()
        off_1.append(r*(p1_r[i] - p2_r[i]) + p1_r[i])
        r = random_state.uniform()
        off_2.append(r*(p4_r[i] - p3_r[i]) + p4_r[i])
    return off_1, off_2

def parametrized_ann(ann_i):
  def ann_ff(weights):
    return ann_i.stimulate(weights)
  return ann_ff


def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection


def roulette_wheel(population, minimization, random_state):
    fitness_cases = np.array([np.power(ind.fitness,2) for ind in population])
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_cases)
        fitness_proportions = np.divide(fitness_cases, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]


def pool_selection(population, minimization, random_state):
    fitness_cases = np.array([ind.fitness for ind in population])
    fitness_cases.sort()
    if minimization:
        pass
    else:
        fit2 = []
        for i in range(len(fitness_cases)):
            fit2 += list(np.repeat(fitness_cases[i], i))
        random_value = random_state.randint(len(fit2))
        for i in range(len(population)):
            if population[i].fitness == fit2[random_value]:
                return population[i]


def stady_selection(population, minimization, random_state):
    sel_df = pd.DataFrame(population)
    fitness_cases = np.array([ind.fitness for ind in population])
    sel_df['fit'] = fitness_cases
    leng = len(fitness_cases)
    # random_value = random_state.randint(leng)
    if minimization:
        pass
    else:
        sel_df = sel_df.sort_values('fit', ascending=False)
        sel_df.drop(sel_df.tail(np.floor(leng * 0.75).astype(int)).index, inplace=True)
        offspring = np.array(sel_df[0])
        fitness_cases = np.array([np.power(ind.fitness,2) for ind in offspring])
        total_fit = np.sum(fitness_cases)
        fitness_proportions = np.divide(fitness_cases, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return offspring[indexes[0][0]]
        # pop = roulette_wheel(offspring, True, random_state)
    # return roulette_wheel(offspring, True, random_state)
    # return np.random.choice(offspring)

def parametrized_tournament_selection2(pressure):
    def stady_selection_tournament(population, minimization, random_state):
        sel_df = pd.DataFrame(population)
        fitness_cases = np.array([ind.fitness for ind in population])
        sel_df['fit'] = fitness_cases
        leng = len(fitness_cases)
        # random_value = random_state.randint(leng)
        if minimization:
            pass
        else:
            sel_df = sel_df.sort_values('fit', ascending=False)
            sel_df.drop(sel_df.tail(np.floor(leng * 0.75).astype(int)).index, inplace=True)
            offspring = np.array(sel_df[0])
            tournament_pool_size = int(len(population) * pressure)
            tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
            if minimization:
                return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
            else:
                return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return stady_selection_tournament


def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection

def plot_acc(arr):
    plt.figure(figsize = (10, 10))
    plt.plot(arr, label = 'Accuracy')
    plt.xlabel('Generation')
    plt.legend(loc = 'best')
    plt.ylim([0, 1])
    plt.title('Accuracy of GA')
    plt.show()

def linearScaling(population):
    fitness_raw = np.array([ind.fitness for ind in population])
    a = max(fitness_raw)
    b = -min(fitness_raw)/np.std(fitness_raw)*2
    fitness_scaled = np.array([a + ind.fitness*b for ind in population])
    return fitness_scaled

def parametrizedRankScalling(preassure):
    def rankScaling(population):
        fitness_raw = np.array([ind.fitness for ind in population])

        # preassure = best/median ratio
        # fitness_scaled = np.array([a + ind.fitness*b for ind in population])
        return fitness_scaled
    return rankScaling

def expScaling(population):
    fitness_raw = np.array([ind.fitness for ind in population])
    # somehow we want to use ranking as in rank scalling
    # fitness_scaled = np.array([a + ind.fitness*b for ind in population])
    return fitness_scaled

# make sure to implement s*N as new fitness, for r>=c

def parametrizedTopScalling(preassure):
    def topScaling(population):
        sel_df = pd.DataFrame(population)
        fitness_raw = np.array([ind.fitness for ind in population])
        sel_df['fit'] = fitness_raw
        leng = len(fitness_raw)
        sel_df = sel_df.sort_values('fit', ascending=False)
        for i in range(leng):
            if i < np.floor(preassure*leng):
                sel_df['fit'][i] = 1
            else: sel_df['fit'][i] = 0
        fitness_scaled = np.array(sel_df[0])
        return fitness_scaled
    return topScaling



def sigmaScaling(population):
    fitness_raw = np.array([ind.fitness for ind in population])
    avg_fit = np.average(fitness_raw)
    std_fit = np.std(fitness_raw)
    fitness_scaled = np.array([1 + ind.fitness - avg_fit/2*std_fit for ind in population])
    return fitness_scaled

def roulette_wheel_fit_scal(population, minimization, random_state):
    fitness_cases = sigmaScaling(population)
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_cases)
        fitness_proportions = np.divide(fitness_cases, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]

def BoltzmanSelection(population, minimization, random_state):
    fitness_raw = np.array([ind.fitness for ind in population])
    control_parameter = 10
    update_rate = 0.9
    avg_fit = np.average(fitness_raw)

    for ind in range(len(fitness_raw)):
        fitness_raw[ind] = np.exp(fitness_raw[ind]/ control_parameter)/ np.exp(avg_fit/ control_parameter)
        control_parameter *= update_rate
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_raw)
        fitness_proportions = np.divide(fitness_raw, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]


def rank_selection(population, minimization, random_state):
    fitness_cases = np.array([ind.fitness for ind in population])
    fitness_cases.sort()
    max = 1.1      # can be < 1 , 2 > Baker recommendation is 1.1 -> from p.170 "intro to GA" Mitchel
    min = 2 - max
    for ind in range(len(fitness_cases)):
        fitness_scaled = np.array([min + (max - min)*(ind - 1) / (len(fitness_cases) - 1) for ind in population])
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_scaled)
        fitness_proportions = np.divide(fitness_scaled, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]