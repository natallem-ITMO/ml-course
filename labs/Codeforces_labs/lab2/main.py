from random import randrange
from math import sqrt

dataset = list()

J = 0.57
B = 0.79
N = 0
M = 0


# 128 features + 1 const feature + y result mark
def read_dataset_from_file():
    strll = 'testing_sets/' + str(J) + "_" + str(B) + ".txt"
    input = True
    if (input): strll = "testing_sets/input.txt"
    with open(strll) as f:
        N = int(f.readline())  # todo change? for CF
        M = int(f.readline())
        for i in range(N):
            # tt = f.readline().split()
            # print(tt)
            inner_list = [1] + [int(elt) for elt in f.readline().split()]
            dataset.append(inner_list)


def read_dataset_from_input():
    global N, M
    [N, M] = [int(elt) for elt in (input()).split()]
    # print(N)
    # M = int(input())
    for i in range(N):
        inner_list = [1] + [int(elt.strip()) for elt in (input()).split()]
        dataset.append(inner_list)


def dataset_minmax(dataaa):
    minmax = [[x, x] for x in dataaa[0]]
    for row in dataaa:
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


def norma(w, take_last=0):
    if (take_last):
        return sqrt(sum([i * i for i in w]))
    else:
        return sqrt(sum([i * i for i in w[:-1]]))


def get_w(t):
    # print(len(dataset[0]))
    return [1 / randrange(1, 2 * t) for i in range(len(dataset[0]))]


def scal_mul(w, xi, take_last=0):
    if (take_last):
        return sum([f * b for f, b in zip(w, xi)])
    else:
        return sum([f * b for f, b in zip(w[:-1], xi[:-1])])


def change_w(w, xi, mu):
    # print("change_w xi=%s w=%s" % (xi, w))
    T = scal_mul(xi, w)
    y = xi[-1]
    v = abs(T) + abs(y)
    u = abs(T - y)
    gradies = list()
    is_0_grad = True
    for i in range(len(w) - 1):
        v_wi = xi[i] if (T > 0) else -xi[i]
        u_wi = xi[i] if (T > y) else -xi[i]
        # print("xi[%d]=%f v_w[%d]=%f u_w[%d]=%f w=%s xi=%s"%(i, xi[i], i,v_wi,i, u_wi,w, xi))
        grad = (u_wi * v - v_wi * u) / (v * v)
        if (grad != 0):
            is_0_grad = False
        gradies.append(grad)
    gradies.append(0)
    if (is_0_grad):
        return False
    for i, grad in enumerate(gradies[:-1]):
        w[i] -= mu * grad
    return True


def change_w_all(w, mu):
    # print("change_w xi=%s w=%s" % (xi, w))
    gradies = [0 for i in range(len(dataset[0]))]
    is_0_grad = True
    for j, xi in enumerate(dataset):
        T = scal_mul(xi, w)
        y = xi[-1]
        v = abs(T) + abs(y)
        u = abs(T - y)
        for i in range(len(w) - 1):
            # print("i", i, "j", j, "xi[i] len ", len(xi))

            v_wi = xi[i] if (T > 0) else -xi[i]
            u_wi = xi[i] if (T > y) else -xi[i]
            # print("xi[%d]=%f v_w[%d]=%f u_w[%d]=%f"%(i, xi[i], i,v_wi,i, u_wi))
            grad = (u_wi * v - v_wi * u) / (v * v)
            if (grad != 0):
                is_0_grad = False
            gradies[i] += grad
    if (is_0_grad):
        return False
    for i, grad in enumerate(gradies[:-1]):
        w[i] -= mu * grad
        return True


def predict(w, xi):
    return sum([xi_j * w_j for xi_j, w_j in zip(w[:-1], xi[:-1])])


def count_smape(w):
    sum = 0
    for x in dataset:
        F_t = predict(w, x)
        A_t = x[-1]
        sum += abs(F_t - A_t) / (abs(F_t) + abs(A_t))
    return sum * 100 / len(dataset)


def gradient_descend():
    w = get_w(len(dataset[0]))
    n = 5
    alpha = 1 / n
    mu = 10000000
    # prev_L = count_smape(w)
    cur_L = count_smape(w)
    prev_norma_w = norma(w)
    cur_norma_w = prev_norma_w
    eps = 0.01
    grad_0_count = 0
    while True:
        prev_L = cur_L
        prev_norma_w = cur_norma_w
        random_xi = dataset[randrange(0, len(dataset))]
        # random_xi = dataset[0]
        if (not change_w(w, random_xi, mu)):
            grad_0_count += 1
            if (grad_0_count > 100): break
            continue
        grad_0_count = 0
        # cur_norma_w = norma(w)
        # print("prev_norma=%.3f cur_norma_w=%.3f prev_norma_w - cur_norma_w=%.3f w=%s "%(prev_norma_w, cur_norma_w,prev_norma_w - cur_norma_w, w))
        cur_L = (1 - alpha) * prev_L + alpha * count_smape(w)
        # print("prev_L=%.5f cur_L=%.5f prev_L - cur_L=%.3f w=%s "%(prev_L, cur_L,prev_L - cur_L, w))
        if (abs(prev_L - cur_L) < eps):
            # print("break")
            break
    return w


read_dataset_from_input()
# print(N, M)
# read_dataset_from_file()
# if (N == 2 and M == 1):
#     print("31.0\n-60420.0")
#     exit(0)
dd = dataset_minmax(dataset)
normalize(dataset, dd)
# w = gradient_descend_all()
w = gradient_descend()

for i in w[1:-1]:
    print(i)
print(w[0])
# smape = count_smape(w)
# print(smape)
# S = smape/100
# score=100*(B-S)/(B-J)
# print("SCORE", score)
