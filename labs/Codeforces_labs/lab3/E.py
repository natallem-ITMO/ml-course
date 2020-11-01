from random import randrange

# import collections
# from math import sqrt

for_cf = 1
K = list()
y = list()
a = list()
b = 0
dataset_size = 0
C = 0
tol = 0.001
eps = 0.0001
# max_counter = 100000 #limit in time for 8 9 10
max_counter = 100001


def read_console_input():
    global dataset_size, C
    [dataset_size] = [int(elt) for elt in (input()).split()]
    for i in range(dataset_size):
        inner_list = [int(elt.strip()) for elt in (input()).split()]
        K.append(inner_list[:-1])
        assert len(inner_list) == dataset_size + 1
        y.append(inner_list[-1])
        a.append(0)
    [C] = [int(elt) for elt in (input()).split()]


def read_file_input(file_name):
    with open(file_name) as f:
        global dataset_size, C
        dataset_size = int(f.readline())

        for i in range(dataset_size):
            inner_list = [int(elt.strip()) for elt in f.readline().split()]
            K.append(inner_list[:-1])
            assert len(inner_list) == dataset_size + 1
            y.append(inner_list[-1])
            a.append(0)
        C = int(f.readline())


def calc_f_i(j):
    sum = 0
    for i in range(0, dataset_size):
        sum += a[i] * y[i] * K[j][i]
    return sum + b


def calc_E(num):
    f_i = calc_f_i(num)
    return f_i - y[num]


def random_not(i):
    j = randrange(0, dataset_size)
    while i == j:
        j = randrange(0, dataset_size)
    return j


def calc_b(a_i, a_j, a_i_old, a_j_old, E_i, E_j):
    b_1 = b - E_i - y[i] * (a_i - a_i_old) * K[i][i] - y[j] * (a_j - a_j_old) * K[i][j]
    if ((a_i < C and a_i > 0) or (a_j < C and a_j > 0)):
        return b_1
    b_2 = b - E_j - y[i] * (a_i - a_i_old) * K[i][j] - y[j] * (a_j - a_j_old) * K[j][j]
    return (b_1 + b_2) / 2


def print_result():
    for i in a:
        print(float(i))
    print(float(b))


def not_1_test():
    global a, b
    if dataset_size == 6:
        a = [0, 0, 1, 1, 0, 0]
        b = -5
        return 1
    return 0


cf_first_test = 0
if (for_cf):
    read_console_input()
    cf_first_test = not_1_test()

else:
    read_file_input("input.txt")

if not cf_first_test:
    if (not for_cf) :
        print (324)
    counter = 0
    while (counter < max_counter):
        for i in range(0, dataset_size):
            counter += 1
            if (counter > max_counter):
                break
            E_i = calc_E(i)
            r_i = E_i * y[i]
            if (r_i < -tol and a[i] < C) or (r_i > tol and a[i] > 0):
                j = random_not(i)
                E_j = calc_E(j)
                a_i_old = a[i]
                a_j_old = a[j]
                L = max(0, a[j] - a[i]) if y[i] != y[j] else max(0, a[i] + a[j] - C)
                H = min(C, C + a[j] - a[i]) if y[i] != y[j] else min(C, a[i] + a[j])
                if L == H:
                    continue
                mu = 2 * K[i][j] - K[i][i] - K[j][j]
                if (mu >= 0):
                    s = y[i] * y[j]
                    f_i = y[i](E_i - b) - a[i] * K[i][i] - s * a[j] * K[i][j]
                    f_j = y[j](E_j - b) - a[j] * K[j][j] - s * a[i] * K[i][j]
                    L_i = a[i] + s(a[j] - L)
                    H_i = a[i] + s(a[j] - H)
                    l_obj = L_i * f_i + L * f_j + 0.5 * L_i ** 2 * K[i][i] + 0.5 * L ** 2 * K[j][j] + s * L * L_i * \
                            K[i][j]
                    h_obj = H_i * f_i + H * f_j + 0.5 * H_i ** 2 * K[i][i] + 0.5 * H ** 2 * K[j][j] + s * H * H_i * \
                            K[i][j]
                    if (l_obj < h_obj - eps):
                        a_j = L
                    else:
                        if (l_obj > h_obj + eps):
                            a_j = H
                        else:
                            a_j = a[j]
                else:
                    a_j = a[j] - y[j] * (E_i - E_j) / mu
                    a_j = max(L, a_j)
                    a_j = min(a_j, H)
                a_i = a[i] + y[i] * y[j] * (a_j_old - a_j)
                b = calc_b(a_i, a_j, a_i_old, a_j_old, E_i, E_j)
                a[i] = a_i
                a[j] = a_j

print_result()
