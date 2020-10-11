from random import randrange
from math import sqrt
from matplotlib import pyplot as plt

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


def draw_plot():
    plt.figure(figsize=(30, 20))

    w_list = package_gradient_descend(train_dataset, iteration_number, mu, tau, package_size)
    print("calculated w_list result w =", w_list[-1])

    mod = 100
    x_part = [i for i in range(1, iteration_number + 1) if i % mod == 0]

    y_train = predict_for(train_dataset, w_list, mod)
    print("calculated y_train")
    plt.plot(x_part, y_train, '.-y', alpha=0.6, label="train dataset", lw=5)

    y_test = predict_for(test_dataset, w_list, mod)
    print("calculated y_test")
    plt.plot(x_part, y_test, '.-r', alpha=0.6, label="test dataset", lw=5)

    plt.legend()
    plt.show()


def test():
    mod = int(iteration_number / 10)
    print("mod", mod)
    w_list = package_gradient_descend(train_dataset, iteration_number, mu, tau, package_size)
    print("calculated w_list")
    y_train = predict_for(train_dataset, w_list, mod)
    y_test = predict_for(test_dataset, w_list, mod)
    print(y_train, "\n", y_test)


min_max_train = dataset_minmax(train_dataset)
min_max_test = dataset_minmax(test_dataset)

dif_y_train = min_max_train[-1][1] - min_max_train[-1][0]
dif_y_test = min_max_test[-1][1] - min_max_test[-1][0]

normalize(train_dataset, min_max_train)
normalize(test_dataset, min_max_test)

iteration_number = 50000
tau = 0.4
n = 20
a = 1 / n
mu = 7
package_size = 1
mod = int(iteration_number / 10)

draw_plot()
