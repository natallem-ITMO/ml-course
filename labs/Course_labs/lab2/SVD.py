from random import randrange
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt

# from numpy.linalg import svd
# from numpy.random import randn

train_dataset = list()
test_dataset = list()

features_num = 0
train_size = 0
test_size = 0
file_name = "testing_sets/2.txt"

# asymptotics for 1 iteration for algos:
# number for grad descend m => m ( / n m )
# number for LSM n*m*m => 1


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


def package_change_w(w, xs, mu, tau):
    gradients = [0 for i in range(len(xs[0]))]
    for vec in xs:
        vec_scal = scal_mul(w, vec)
        y = vec[-1]
        sign = 1 if (vec_scal > y) else -1
        for i, j in enumerate(vec):
            gradients[i] += j * sign
    for i in range(0, len(gradients)):
        gradients[i] /= len(xs)
    w = [x * (1 - mu * tau) - mu * grad for (x, grad) in zip(w, gradients)]
    return w


def predict(w, x):
    return sum([xi_j * w_j for xi_j, w_j in zip(w[:-1], x[:-1])])


def package_gradient_descend(dataset, iteration_number, mu, tau, package_size):
    w = get_w(len(dataset[0]))
    w_list = [w]
    for i in range(iteration_number):
        rand_xs = [dataset[randrange(0, len(dataset))] for i in range(0, package_size)]
        w = package_change_w(w, rand_xs, mu / (iteration_number * 2), tau)
        w_list.append(w)
    return w_list[:-1]


def predict_for(dataset, w_list, mod):
    y = [calc_NRMSE(w_list[0], dataset)]
    for i, w in enumerate(w_list):
        if (i % mod != 0):
            continue
        L_k_prev = y[(i % mod) - 1]
        L_cur = calc_NRMSE(w, dataset)
        y.append((1 - a) * L_k_prev + a * L_cur)
    return y[1:]


min_max_train = dataset_minmax(train_dataset)
min_max_test = dataset_minmax(test_dataset)

dif_y_train = min_max_train[-1][1] - min_max_train[-1][0]
dif_y_test = min_max_test[-1][1] - min_max_test[-1][0]

normalize(train_dataset, min_max_train)
normalize(test_dataset, min_max_test)

numpy_array_1 = np.array([[40], [23]])
numpy_array_2 = np.array([[43], [223]])


# print(numpy_array_1 * 3)
# print(numpy_array_1.shape,numpy_array_2.shape)

def calc_tetas():
    n = features_num
    start2 = np.ones_like(U_t[0])
    # start2 = np.array([[1] for i in range(features_num)])
    # print(start2.shape)
    # print(start2)
    # print(V[1].transpose().shape)
    for i in range(0, features_num):
        # print(1/s[i])
        alpha = s[i] / (s[i] ** 2 + tau)
        # print (alpha)
        start2 += alpha * U_t[0] * (V_t[0].dot(y))
        # print(i, start2.tolist())
    # print("V.shape", (U_t[0]* (V_t[0].dot(y)) ).shape)
    # print("V.shape", (U_t[0]* (V_t[0].dot(y)) + start2))
    # for  i in range(0,n):
    #     print(i)
    return start2


tau = 25
F = [y[:-1] for y in train_dataset]
y = [[y[-1]] for y in train_dataset]
print(len(train_dataset), len(F), len(train_dataset[0]), len(F[0]), len(y), len(y[0]))

u, s, vh = np.linalg.svd(F, full_matrices=False)
V = u
ll = [(i * i) / i for i in s]
# print("s", s.tolist())
# print("ll", ll)
# print("diag my", np.diag(ll).tolist())
# print("diag orig", np.diag(s).tolist())
D = np.diag(s)
my_D = np.diag(ll)
U_t = vh
U = U_t.transpose()

D_1 = np.linalg.inv(D)
my_D_1 = np.linalg.inv(my_D)
V_t = V.transpose()
n = features_num
# print(features_num)
# print("U shape", U.shape)
# te = calc_tetas()
# tetas_t = U.dot(D_1).dot(V_t).dot(y)
# tetas = (tetas_t.transpose()[0]).tolist() + [0]
# tetas2 = calc_tetas().tolist() + [0]

list_tau = ([tau for i in range(0, features_num)])
tau_I_n = np.diag(list_tau)
instead_of_D_1 = np.linalg.inv(D.dot(D) + tau_I_n).dot(D)
my_tetas_t = U.dot(instead_of_D_1).dot(V_t).dot(y)
my_tetas = (my_tetas_t.transpose()[0]).tolist() + [0]
my_tetas2 = calc_tetas().tolist() + [0]

# print(len(tetas), "Tetas", tetas)
# print(calc_NRMSE(tetas, train_dataset))
# print(calc_NRMSE(tetas, test_dataset))
print(len(my_tetas), "Tetas my", my_tetas)
print(calc_NRMSE(my_tetas, train_dataset))
print(calc_NRMSE(my_tetas, test_dataset))

# F_t = F.transpose()
# tetas2_t= np.linalg.inv(F_t.dot(F)).dot(F_t).
# tetas2= (tetas_t.transpose()[0]).tolist() + [0]

# a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
# a= [[3,4,3],[3,2,3], [3,2,3],[2,3,21]]
# u, s, vh = np.linalg.svd(a, full_matrices=False)
# print(np.diag(s))
# print(np.array(a).transpose().tolist())
# print(u.shape, s.shape, vh.shape)
# print(u.dot(np.diag(s)).dot(vh)[3][2])
