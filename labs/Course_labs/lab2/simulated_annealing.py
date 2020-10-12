from random import randrange
from random import random
from math import sqrt
from math import exp
from matplotlib import pyplot as plt

train_dataset = list()
test_dataset = list()

features_num = 0
train_size = 0
test_size = 0
file_name = "testing_sets/2.txt"

with open(file_name) as f:
    features_num = int(f.readline())
    train_size = int(f.readline())
    for i in range(train_size):
        inner_list = [int(elt) for elt in f.readline().split()]
        if (len(inner_list) != (features_num + 1)):
            continue
        train_dataset.append(inner_list)
    test_size = int(f.readline())
    for i in range(test_size):
        inner_list = [int(elt) for elt in f.readline().split()]
        if (len(inner_list) != (features_num + 1)):
            continue
        test_dataset.append(inner_list)


def dataset_minmax(dataset):
    minmax = [[x, x] for x in dataset[0]]
    for row in dataset:
        for idx, x in enumerate(row):
            minmax[idx][0] = min(minmax[idx][0], x)
            minmax[idx][1] = max(minmax[idx][1], x)
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if (minmax[i][1] - minmax[i][0] == 0):
                row[i] = 1
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def scal_mul(w, xi, take_last=0):
    if (take_last):
        return sum([f * b for f, b in zip(w, xi)])
    else:
        return sum([f * b for f, b in zip(w[:-1], xi[:-1])])


def norma(w, take_last=0):
    if (take_last):
        return sqrt(sum([i * i for i in w]))
    else:
        return sqrt(sum([i * i for i in w[:-1]]))


def get_w(t):
    return [(1 / randrange(1, 2 * t)) * 0 for i in range(features_num)]


def calc_T(w, dataset):
    cur_sum = 0
    for vec in dataset:
        t = (scal_mul(w, vec) - vec[-1])
        t *= t
        cur_sum += t
    return cur_sum


def calc_MSE(w, dataset):
    return calc_T(w, dataset) / len(dataset)


def calc_NRMSE(w, dataset):
    return sqrt(calc_MSE(w, dataset))


def draw_plot():
    plt.figure(figsize=(30, 20))
    (x, y_train, y_test) = simulated_annealing(train_dataset, start_temp, end_temp)
    print("Calculated.\nNRMSE for train dataset:", y_train[-1], "\nNRMSE for test dataset:", y_test[-1])
    plt.plot(x, y_train, '.-y', alpha=0.6, label="train dataset", lw=5)
    plt.plot(x, y_test, '.-r', alpha=0.6, label="test dataset", lw=5)
    plt.show()


def gen_w(w):
    index = randrange(0, len(w) - 1)
    w_new = w.copy()
    sign = -1 if (random() >= 0.5) else 1
    step = random() * coeff_step_gen
    w_new[index] = w_new[index] + sign * step
    return w_new


def get_transition_probability(dE, T):
    return exp(-dE / T)


def is_transition(probability):
    value = random()
    return 1 if (value <= probability) else 0


def decrease_temperature(initial_temp, i):
    return initial_temp * 0.1 / i


def simulated_annealing(dataset, initial_temperature, end_temperature):
    x = list()
    train_nrmse = list()
    test_nrmse = list()

    w = get_w(len(dataset[0]))
    current_nrmse = calc_NRMSE(w, dataset)
    current_energy = current_nrmse + norma(w)
    T = initial_temperature
    for i in range(1, iteration_number):
        x.append(i)
        train_nrmse.append(current_nrmse)
        test_nrmse.append(calc_NRMSE(w, test_dataset))
        w_candidate = gen_w(w)
        candidate_nrmse = calc_NRMSE(w_candidate, dataset)
        candidate_energy = candidate_nrmse + norma(w_candidate) * tau

        if candidate_energy < current_energy:
            current_energy = candidate_energy
            current_nrmse = candidate_nrmse
            w = w_candidate
        else:
            p = get_transition_probability(candidate_energy - current_energy, T)
            if is_transition(p):
                current_nrmse = candidate_nrmse
                current_energy = candidate_energy
                w = w_candidate
        T = decrease_temperature(initial_temperature, i)
        if T <= end_temperature:
            break
    return (x, train_nrmse, test_nrmse)


min_max_train = dataset_minmax(train_dataset)
min_max_test = dataset_minmax(test_dataset)

normalize(train_dataset, min_max_train)
normalize(test_dataset, min_max_test)

iteration_number = 8000
start_temp = 10
end_temp = 0.00001
tau = 0.07
coeff_step_gen = 0.01
draw_plot()
