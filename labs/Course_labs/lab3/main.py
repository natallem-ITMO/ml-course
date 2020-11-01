import pandas as pd
import math
import random
from random import randrange

from matplotlib import pyplot as plt


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = -1 if i == 0 else 1
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def read_and_prepare_dataset(dataset_name):
    dataset_ = pd.read_csv("datasets/" + dataset_name)
    # print(dataset_)
    dataset = dataset_.values.tolist()
    str_column_to_int(dataset, len(dataset[0]) - 1)
    return dataset


def cacl_K(dataset, core_func):
    empty_str = [0].copy() * len(dataset)
    matrix = [empty_str.copy()] * len(dataset)
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            matrix[i][j] = core_func(dataset[i], dataset[j])
    return matrix


def liniar(vec1, vec2):
    sum = 0
    for i, j in zip(vec1[:-1], vec2[:-1]):
        sum += i * j
    return sum


def calc_b(a_i, a_j, a_i_old, a_j_old, E_i, E_j, b, y, K, i, j):
    b_1 = b - E_i - y[i] * (a_i - a_i_old) * K[i][i] - y[j] * (a_j - a_j_old) * K[i][j]
    if ((a_i < C and a_i > 0) or (a_j < C and a_j > 0)):
        return b_1
    b_2 = b - E_j - y[i] * (a_i - a_i_old) * K[i][j] - y[j] * (a_j - a_j_old) * K[j][j]
    return (b_1 + b_2) / 2


def cacl_f_i(j, b, y, K, a):
    sum = 0
    for i in range(0, len(a)):
        sum += a[i] * y[i] * K[j][i]
    return sum + b


def calc_E(num, b, y, K, a):
    f_i = cacl_f_i(num, b, y, K, a)
    return f_i - y[num]


def random_not(i, dataset_size):
    j = randrange(0, dataset_size)
    while i == j:
        j = randrange(0, dataset_size)
    return j


# def find_a(a, b, train):
#     dataset_size = len(train)
#     K = cacl_K(train, liniar)
#     y = [cur[-1] for cur in train]
#
#     counter = 0
#     while (counter < max_passes):
#         for i in range(0, dataset_size):
#             counter += 1
#             # print(counter)
#             if (counter > max_passes):
#                 break
#             E_i = calc_E(i, b, y, K, a)
#             r_i = E_i * y[i]
#             if (r_i < -tol and a[i] < C) or (r_i > tol and a[i] > 0):
#                 j = random_not(i, dataset_size)
#                 # print(counter, i, j)
#                 E_j = calc_E(j, b, y, K, a)
#                 a_i_old = a[i]
#                 a_j_old = a[j]
#                 L = max(0, a[j] - a[i]) if y[i] != y[j] else max(0, a[i] + a[j] - C)
#                 H = min(C, C + a[j] - a[i]) if y[i] != y[j] else min(C, a[i] + a[j])
#                 if L == H:
#                     continue
#                 mu = 2 * K[i][j] - K[i][i] - K[j][j]
#                 if (mu >= 0):
#                     s = y[i] * y[j]
#                     f_i = y[i]*(E_i - b) - a[i] * K[i][i] - s * a[j] * K[i][j]
#                     f_j = y[j]*(E_j - b) - a[j] * K[j][j] - s * a[i] * K[i][j]
#                     L_i = a[i] + s*(a[j] - L)
#                     H_i = a[i] + s*(a[j] - H)
#                     l_obj = L_i * f_i + L * f_j + 0.5 * L_i ** 2 * K[i][i] + 0.5 * L ** 2 * K[j][j] + s * L * L_i * \
#                             K[i][j]
#                     h_obj = H_i * f_i + H * f_j + 0.5 * H_i ** 2 * K[i][i] + 0.5 * H ** 2 * K[j][j] + s * H * H_i * \
#                             K[i][j]
#                     if (l_obj < h_obj - eps):
#                         a_j = L
#                     else:
#                         if (l_obj > h_obj + eps):
#                             a_j = H
#                         else:
#                             a_j = a[j]
#                 else:
#                     a_j = a[j] - y[j] * (E_i - E_j) / mu
#                     a_j = max(L, a_j)
#                     a_j = min(a_j, H)
#                 a_i = a[i] + y[i] * y[j] * (a_j_old - a_j)
#                 print("changed a", a_i, a_i_old, a_j, a_j_old)
#                 b = calc_b(a_i, a_j, a_i_old, a_j_old, E_i, E_j, b, y, K,i, j)
#                 a[i] = a_i
#                 a[j] = a_j


def find_a_b(train):
    a = [0].copy() * len(train)
    b = 0
    dataset_size = len(train)
    K = cacl_K(train, cur_core_function)
    y = [cur[-1] for cur in train]

    passes = 0
    while (passes < max_passes):
        num_changed_alphas = 0
        for i in range(0, dataset_size):

            # passes += 1
            # print(passes)
            if (passes > max_passes):
                break
            E_i = calc_E(i, b, y, K, a)
            r_i = E_i * y[i]
            if (r_i < -tol and a[i] < C) or (r_i > tol and a[i] > 0):
                j = random_not(i, dataset_size)
                # print(passes, i, j)
                E_j = calc_E(j, b, y, K, a)
                a_i_old = a[i]
                a_j_old = a[j]
                L = max(0, a[j] - a[i]) if y[i] != y[j] else max(0, a[i] + a[j] - C)
                H = min(C, C + a[j] - a[i]) if y[i] != y[j] else min(C, a[i] + a[j])
                if L == H:
                    continue
                mu = 2 * K[i][j] - K[i][i] - K[j][j]
                if (mu >= 0):

                    continue
                    s = y[i] * y[j]
                    f_i = y[i] * (E_i - b) - a[i] * K[i][i] - s * a[j] * K[i][j]
                    f_j = y[j] * (E_j - b) - a[j] * K[j][j] - s * a[i] * K[i][j]
                    L_i = a[i] + s * (a[j] - L)
                    H_i = a[i] + s * (a[j] - H)
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
                if (abs(a_j - a_j_old) < eps):
                    continue
                a_i = a[i] + y[i] * y[j] * (a_j_old - a_j)
                print("changed a for ", i,j, "passes", passes, a_i, a_i_old, a_j, a_j_old)
                print("now accuracy =", cacl_accuracy(a,b,dataset,dataset,cur_core_function))
                b = calc_b(a_i, a_j, a_i_old, a_j_old, E_i, E_j, b, y, K, i, j)
                a[i] = a_i
                a[j] = a_j
                num_changed_alphas += 1
        if (num_changed_alphas == 0):
            print(passes, "+1")
            passes += 1
        else:
            passes = 0
    return [a, b]


def predict_for(row, a, b, dataset, core_function, need_zero=0):
    sum = 0
    for i in range(0, len(a)):
        sum += a[i] * dataset[i][-1] * core_function(dataset[i], row)
    sum += b
    result = -1
    if sum == 0 and need_zero:
        result = 0
    else:
        if sum > 0:
            result = 1
    return result


def cacl_accuracy(a, b, train, test, core_function):
    correct_predicted = 0
    for row in test:
        result = predict_for(row, a, b, train, core_function)
        if (result == row[-1]):
            correct_predicted += 1
    return correct_predicted / len(test)


def get_bounds(dataset):
    min_width = dataset[0][0]
    max_width = dataset[0][0]
    min_height = dataset[0][1]
    max_height = dataset[0][1]
    for row in dataset:
        min_width = min(min_width, row[0])
        max_width = max(max_width, row[0])
        min_height = min(min_height, row[1])
        max_height = max(max_height, row[1])
    return [min_width, max_width, min_height, max_height]


def add_margin(min_width, max_width, min_height, max_height):
    margin = 0.2
    x_start = min_width - (max_width - min_width) * margin
    x_end = max_width + (max_width - min_width) * margin
    y_start = min_height - (max_height - min_height) * margin
    y_end = max_height + (max_height - min_height) * margin
    return [x_start, x_end, y_start, y_end]


def show_pic(data_pic):
    fig = plt.figure(figsize=(30, 30))
    pc = plt.pcolor(data_pic)  # метод псевдографики pcolor
    plt.title('Hello')
    plt.show()


def drow_dataset(dataset, a, b):
    mark_class_P = 100
    mark_class_N = -100
    mark_all_P = 200
    mark_all_N = -200
    mark_border = 400

    [min_width, max_width, min_height, max_height] = get_bounds(dataset)
    [x_start, x_end, y_start, y_end] = add_margin(min_width, max_width, min_height, max_height)
    x_start = min(x_start, y_start)

    if (x_start < 0):
        x_start = math.floor(x_start)
    else:
        x_start = math.ceil(x_start)
    y_start = x_start
    x_end = max(x_end, y_end)

    if (x_end < 0):
        x_end = math.floor(x_end)
    else:
        x_end = math.ceil(x_end)
    y_end = x_end

    print(min_width, max_width, min_height, max_height)
    print(x_start, x_end, y_start, y_end)

    size = 400
    s = (x_end - x_start) / size
    data_pic = list()
    # print("datapic",data_pic)
    for i in range(size):
        cur_line = list()
        y = y_start + s * i
        for j in range(size):
            print(i, j)
            x = x_start + s * j
            cur_prediction = predict_for([x, y, 1], a, b, dataset, cur_core_function, 1)
            if cur_prediction == 0:
                cur_line.append(mark_border)
            else:
                if cur_prediction == 1:
                    cur_line.append(mark_all_P)
                if cur_prediction == -1:
                    cur_line.append(mark_all_N)
        data_pic.append(cur_line)
    for row in dataset:
        x_row = row[0]
        y_row = row[1]
        k_x = math.floor((x_row - x_start) / s)
        k_y = math.floor((y_row - y_start) / s)
        cur_color = mark_class_N if row[2] == -1 else mark_class_P
        data_pic[k_y][k_x] = cur_color

    show_pic(data_pic)


def test_on_blocks():
    size_of_block = math.ceil(len(dataset) / num_blocks)
    random.shuffle(dataset)
    start = 0
    end = start + size_of_block
    sum_accuracy = 0
    for i in range(num_blocks):
        train_dataset = list()
        test_dataset = list()
        for num, j in enumerate(dataset):
            if num in range(start, end):
                test_dataset.append(j)
            else:
                train_dataset.append(j)
        start = end
        end += size_of_block
        [a, b] = find_a_b(train_dataset)
        ress = cacl_accuracy(a, b, train_dataset, test_dataset, cur_core_function)
        print("I=", i, "accuracy=", ress)
        sum_accuracy += ress
    return sum_accuracy / num_blocks


# PARAMETERS
num_blocks = 5
tol = 0.0000001
eps = 0.000001
max_passes = 500
cur_core_function = liniar

# HYPER PARAMETERS
C_choice = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
C = C_choice[-1]

dataset = read_and_prepare_dataset("geyser.csv")
# dataset = read_and_prepare_dataset("chips.csv")
# print("average acc", test_on_blocks())

[a, b] = find_a_b(dataset)

print(cacl_accuracy(a, b, dataset, dataset, cur_core_function))

drow_dataset(dataset, a, b)
