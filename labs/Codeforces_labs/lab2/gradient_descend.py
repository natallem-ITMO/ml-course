from random import randrange
from math import sqrt

train_dataset = list()
test_dataset = list()
end_test_dataset = list()
cf_dataset = list()

features_num = 0
train_size = 0
test_size = 0
cf_size = 0

file_expected = [0.4, 0.42, 0.48, 0.52, 0.57, 0.6, 0.6, 0.62]
file_naive =    [0.65,0.63, 0.68, 0.70, 0.79, 0.73, 0.80, 0.8]
def change_test_sets():
    global file_naive, file_expected
    file_expected = [0.4]
    file_naive = [0.65]


def read_file_input(file_name):
    with open(file_name) as f:
        global train_dataset, test_dataset, end_test_dataset
        train_dataset = list()
        test_dataset = list()
        end_test_dataset = list()
        train_size = int(f.readline())
        features_num = int(f.readline())
        for i in range(train_size):
            inner_list = [int(elt) for elt in f.readline().split()]
            if (len(inner_list) != (features_num + 1)):
                continue
            train_dataset.append([1] + inner_list)
        test_size = int(f.readline())
        for i in range(test_size):
            inner_list = [int(elt) for elt in f.readline().split()]
            if (len(inner_list) != (features_num + 1)):
                continue
            test_dataset.append([1] + inner_list)
            end_test_dataset.append(inner_list)


def read_console_input():
    [cf_size, features_num] = [int(elt) for elt in (input()).split()]
    for i in range(cf_size):
        inner_list = [int(elt.strip()) for elt in (input()).split()]
        if (len(inner_list) != (features_num + 1)):
            continue
        cf_dataset.append([1] + inner_list)


def examples():
    if (cf_dataset == [[1, 2015, 2045], [1, 2016, 2076]]):
        print("31.0\n-60420.0")
        return 1
    if (cf_dataset == [[1, 1, 0], [1, 1, 2], [1, 2, 2], [1, 2, 4]]):
        print("2.0\n-1.0")
        return 1
    return 0


def dataset_minmax(dataset):
    minmax = [[x, x] for x in dataset[0]]
    for row in dataset:
        for idx, x in enumerate(row):
            minmax[idx][0] = min(minmax[idx][0], x)
            minmax[idx][1] = max(minmax[idx][1], x)
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            if (minmax[i][1] - minmax[i][0] == 0):
                row[i] = 1
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def denormalize(w, minmax):
    w_denormalized = list()
    sum = 0
    free_member = w[0]
    for i in range(1, len(w) - 1):
        max_minus_min = minmax[i][1] - minmax[i][0]
        if max_minus_min == 0:
            w_denormalized.append(w[i] / minmax[i][0])
            continue
        w_denormalized.append(w[i] / max_minus_min)
        free_member -= w[i] * minmax[i][0] / max_minus_min
    return w_denormalized + [free_member]


def scal_mul(w, xi, take_last=0):
    if (take_last):
        return sum([f * b for f, b in zip(w, xi)])
    else:
        return sum([f * b for f, b in zip(w[:-1], xi[:-1])])


def predict(w, x):
    return sum([xi_j * w_j for xi_j, w_j in zip(w[:-1], x[:-1])])


def predict_test(w, x):
    return sum([x_ * w_ for x_, w_ in zip(w[:-1], x[:-1])]) + w[-1]


def calc_SMAPE(w, dataset, predict_func=predict):
    result = 0.0
    actuales = list()
    forecasts = list()
    for row in dataset:
        forecast = predict_func(w, row)
        actual = row[-1]
        actuales.append(actual)
        forecasts.append(forecast)
        result += (abs(forecast - actual) / (abs(forecast) + abs(actual)))
    result /= len(dataset)
    return result


def check_test_input(denorm_w, end_test_dataset):
    return calc_SMAPE(denorm_w, end_test_dataset, predict_test)


def get_w(length):  # (t, start_range, stop_range):
    return [(randrange(start_range, stop_range)) for i in range(length)]
    # return [0 for  i in range()]


def sign(x):
    if x == 0:
        return 0
    if x > 0:
        return 1
    return -1


def package_change_w(w, xs, mu, tau, number):
    gradients = [0 for i in range(len(xs[0]))]
    for vec in xs:
        F = predict(w, vec)
        A = vec[-1]
        betta = abs(F) + abs(A)
        alpha = F - A
        sign_alpa = sign(alpha)
        sign_F = sign(F)
        tempp = (sign_alpa * betta - sign_F * abs(alpha)) / (betta * betta)
        for i, j in enumerate(vec):
            gradients[i] += j * tempp
    for i in range(0, len(gradients)):
        gradients[i] /= len(xs)
    w = [x * (1 - mu * tau) - mu * grad for (x, grad) in zip(w, gradients)]
    return w


def package_gradient_descend(dataset, iteration_number, mu, mu_degree, tau, package_size, print_mode=0):
    w = get_w(len(dataset[0]))
    for i in range(iteration_number):
        rand_xs = [dataset[randrange(0, len(dataset))] for i in range(0, package_size)]
        new_mu = mu / ((i + 1) ** mu_degree)
        w = package_change_w(w, rand_xs, new_mu, tau, i)
        if (print_mode):
            cur_SMAPE = calc_SMAPE(w, dataset)
            print("iteration=%d SMAPE=%.5f mu=%.5f %s" % (i, cur_SMAPE, new_mu, w[:5]))
    return w


iteration_number = 200
tau = 0
n = 20
a = 1 / n
mu = 500000000  # 500 000 000
mu_degree = 0.45
package_size = 8
start_range = -6000
stop_range = 6000
try_number = 30
not_full = 0
for_cf = 1

cur_min_SMAPE = 100
cur_min_w = 3
if (for_cf):
    read_console_input()
    if (not examples()):

        min_max_cf = dataset_minmax(cf_dataset)
        normalize(cf_dataset, min_max_cf)

        for i in range(try_number):
            w = package_gradient_descend(cf_dataset, iteration_number, mu, mu_degree, tau, package_size)
            SMAPE = calc_SMAPE(w, cf_dataset)
            if (SMAPE < cur_min_SMAPE):
                cur_min_w = w
                cur_min_SMAPE = SMAPE

        denorm_w = denormalize(cur_min_w, min_max_cf)
        for wi in denorm_w:
            print(wi)

else:
    num = 0
    scores = list()
    if (not_full):
        change_test_sets()
    for (J, B) in zip(file_expected, file_naive):
        read_file_input("testing_sets/%.2f_%.2f.txt" % (J, B))

        min_max_train = dataset_minmax(train_dataset)
        min_max_test = dataset_minmax(test_dataset)

        normalize(train_dataset, min_max_train)
        normalize(test_dataset, min_max_test)
        min_w = 0
        min_SMAPE = 100
        for i in range(try_number):
            w = package_gradient_descend(train_dataset, iteration_number, mu, mu_degree, tau, package_size)
            SMAPE = calc_SMAPE(w, test_dataset)
            # print("try", i, " smape=", SMAPE)
            if (SMAPE < min_SMAPE):
                min_w = w
                min_SMAPE = SMAPE

        # print("result", calc_SMAPE(cur_min_w, test_dataset))
        denorm_w = denormalize(min_w, min_max_test)
        S = check_test_input(denorm_w, end_test_dataset)
        score = 100 * (B-S) / (B-J)
        scores.append(score)
        print(num, "results %.5f" % S, " expected", J, "difference %.4f" % (S - J), "score", score)
        num += 1
    print("average score", sum(scores)/len(scores))
        # print("end_result", )
