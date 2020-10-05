from random import randrange
from math import sqrt

study_dataset = list()
training_dataset = list()

features_num = 0
testing_num = 0
study_num = 0
file_name = "testing_sets/4.txt"

with open(file_name) as f:
    features_num = int(f.readline())
    study_num = int(f.readline())
    for i in range(study_num):
        inner_list = [int(elt) for elt in f.readline().split()]
        if (len(inner_list) != (features_num + 1)):
            continue
        study_dataset.append(inner_list)
    testing_num = int(f.readline())
    for i in range(testing_num):
        inner_list = [int(elt) for elt in f.readline().split()]
        if (len(inner_list) != (features_num + 1)):
            continue
        training_dataset.append(inner_list)

print(features_num)


def dataset_minmaxsum(cur_dataset):
    minmaxsum = [[x, x, 0] for x in cur_dataset[0]]
    for row in cur_dataset:
        for idx, x in enumerate(row):
            minmaxsum[idx][0] = min(minmaxsum[idx][0], x)
            minmaxsum[idx][1] = max(minmaxsum[idx][1], x)
            minmaxsum[idx][2] += x
    return minmaxsum


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
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
    # print(len(dataset[0]))
    return [(1 / randrange(1, 2 * t)) + 4 for i in range(len(study_dataset[0]))]


def calc_T(w):
    cur_sum = 0
    for vec in training_dataset:
        t = (scal_mul(w, vec) - vec[-1])
        t *= t
        cur_sum += t
    return cur_sum


def calc_MSE(w):
    return calc_T(w) / len(training_dataset)


def calc_RMSE(w):
    return sqrt(calc_MSE(w))


def calc_NRMSE(w):
    return calc_RMSE(w) / dif_y


def change_w(w, mu, tau):
    # print("change_w xi=%s w=%s" % (xi, w))
    T = calc_T(w)
    T_all = sqrt(T) * sqrt(study_num) * dif_y
    gradients = [min_max_sum[i][2] / T_all for i in range(len(w))]
    # print("w before", w)
    for i, grad in enumerate(gradients[:-1]):
        w[i] = w[i] * (1 - mu * tau) - mu * grad
    # print("w after", w)


def predict(w, x):
    return sum([xi_j * w_j for xi_j, w_j in zip(w[:-1], x[:-1])])


def count_SMAPE(w):
    sum = 0
    for x in study_dataset:
        F_t = predict(w, x)
        A_t = x[-1]
        sum += abs(F_t - A_t) / (abs(F_t) + abs(A_t))
    return sum / len(study_dataset)


def gradient_descend(min_max_sum):
    w = get_w(len(study_dataset[0]))
    tau = 0.01  # (1 - mu * tau)
    mu = 0.1
    grad_0_count = 0
    number_iterations = 10000
    for i in range(number_iterations):
        print("NRMSE before", calc_NRMSE(w))
        change_w(w,mu,tau);
        print("NRMSE after", calc_NRMSE(w))
    return w


min_max_sum = dataset_minmaxsum(study_dataset)
dif_y = min_max_sum[-1][1] - min_max_sum[-1][0]
normalize(study_dataset, min_max_sum)
# w = gradient_descend_all()
w = gradient_descend(min_max_sum)
nrmse = calc_NRMSE(w)
print(nrmse)
# for i in w[1:-1]:
# print(i)
# print(w[0])
