from math import sqrt
import numpy as np

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


min_max_train = dataset_minmax(train_dataset)
min_max_test = dataset_minmax(test_dataset)

normalize(train_dataset, min_max_train)
normalize(test_dataset, min_max_test)

# Regularization parameter
tau = 25

# Create input matrices
F = [y[:-1] for y in train_dataset]
y = [[y[-1]] for y in train_dataset]

# Using SVD and create others matrices, calculate w
u, s, vh = np.linalg.svd(F, full_matrices=False)
V = u
D = np.diag(s)
U_t = vh
U = U_t.transpose()
V_t = V.transpose()
list_tau = ([tau for i in range(0, features_num)])
tau_I_n = np.diag(list_tau)
instead_of_D_1 = np.linalg.inv(D.dot(D) + tau_I_n).dot(D)
w_t = U.dot(instead_of_D_1).dot(V_t).dot(y)
w = (w_t.transpose()[0]).tolist() + [0]

# print result
print("Regularization parameter:", tau)
print("Features num:", features_num, "\nSize of train dataset", train_size, "\nSize of test dataset", test_size)
print("NRMSE for train dataset:", calc_NRMSE(w, train_dataset))
print("NRMSE for test dataset:", calc_NRMSE(w, test_dataset))
