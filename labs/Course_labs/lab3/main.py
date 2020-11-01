import pandas as pd
import math
import random
from random import randrange
from math import exp
from matplotlib import pyplot as plt


def change_C(c):
    global C
    C = c


def change_P(p):
    global P
    P = p


def change_B(b):
    global B
    B = b


param_name_variable_dic = {
    "C": change_C,
    "P": change_P,
    "B": change_B
}


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


def calc_K(dataset):
    empty_str = [0 for i in range(len(dataset))]
    matrix = [empty_str.copy() for i in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            matrix[i][j] = cur_core_function(dataset[i], dataset[j])
    return matrix


def linear_kernel(vec1, vec2):
    sum = 0
    for i, j in zip(vec1[:-1], vec2[:-1]):
        sum += i * j
    return sum


def polynomial_kernel(vec1, vec2):
    return (1 + linear_kernel(vec1, vec2)) ** P


def gaussian_kernel(vec1, vec2):
    return exp(-B * ((vec1[0] - vec2[0]) ** 2 + (vec1[1] - vec2[1]) ** 2))


def calc_f_i(j):
    sum = 0
    for i in range(0, len(a)):
        sum += a[i] * y[i] * K[j][i]
    return sum + b


def calc_E(j):
    sum = 0
    for i in range(len(a)):
        sum += y[i] * a[i] * K[j][i]
    return sum + b - y[j]


def random_not(i):
    j = randrange(0, len(a))
    while i == j:
        j = randrange(0, len(a))
    return j


def calc_b(a_i, a_j, a_i_old, a_j_old, E_i, E_j, i, j):
    b_1 = b - E_i - y[i] * (a_i - a_i_old) * K[i][i] - y[j] * (a_j - a_j_old) * K[i][j]
    b_2 = b - E_j - y[i] * (a_i - a_i_old) * K[i][j] - y[j] * (a_j - a_j_old) * K[j][j]
    return (b_1 + b_2) / 2


def calc_b_2():
    cur_index = -1
    for index, val in enumerate(a):
        if 0 < val < C:
            cur_index = index
            break
    if cur_index == -1:
        return b
    return -(calc_f_i(cur_index) - y[cur_index])


def find_a_b(train, test):
    global y, K, a, b
    a = [0 for i in train]
    b = 0
    y = [x[-1] for x in train]
    K = calc_K(train)
    counter = 0

    indeces = [i for i in range(0, len(train))]
    while counter < max_counter:
        counter += 1
        random.shuffle(indeces)
        for i in range(0, len(train)):
            j = indeces[i]
            if i == j:
                continue
            [E_i, E_j] = [calc_E(i), calc_E(j)]
            r_i = E_i * y[i]
            if (r_i < -tol and a[i] < C) or (r_i > tol and a[i] > 0):
                a_i_old = a[i]
                a_j_old = a[j]
                L = max(0, a[j] - a[i]) if y[i] != y[j] else max(0, a[i] + a[j] - C)
                H = min(C, C + a[j] - a[i]) if y[i] != y[j] else min(C, a[i] + a[j])
                if L == H:
                    continue
                mu = 2 * K[i][j] - K[i][i] - K[j][j]
                if mu == 0:
                    continue
                a_j = a[j] - y[j] * (E_i - E_j) / mu
                a_j = max(L, min(a_j, H))
                a_i = a[i] + y[i] * y[j] * (a_j_old - a_j)
                b = calc_b(a_i, a_j, a_i_old, a_j_old, E_i, E_j, i, j)
                a[i] = a_i
                a[j] = a_j
        if counter % 100 == 0:
            print(counter, calc_accuracy(train, test))


def predict_for(row, dataset):
    sum = 0
    for i in range(0, len(a)):
        sum += a[i] * dataset[i][-1] * cur_core_function(dataset[i], row)
    sum += b
    result = -1
    if sum >= 0:
        result = 1
    return result


def calc_accuracy(train, test):
    correct_predicted = 0
    for row in test:
        result = predict_for(row, train)
        if result == row[-1]:
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


def show_pic(data_pic, name):
    fig = plt.figure(figsize=(30, 30))
    pc = plt.pcolor(data_pic)  # метод псевдографики pcolor
    plt.title(name)
    plt.show()


def drow_dataset(dataset, name):
    mark_class_P = 1000
    mark_class_N = -1000
    mark_all_P = 400
    mark_all_N = -400

    [min_width, max_width, min_height, max_height] = get_bounds(dataset)
    [x_start, x_end, y_start, y_end] = add_margin(min_width, max_width, min_height, max_height)
    x_start = min(x_start, y_start)

    if x_start < 0:
        x_start = math.floor(x_start)
    else:
        x_start = math.floor(x_start)
    y_start = x_start
    x_end = max(x_end, y_end)

    if x_end < 0:
        x_end = math.ceil(x_end)
    else:
        x_end = math.ceil(x_end)
    y_end = x_end

    size = 500
    s = (x_end - x_start) / size
    data_pic = list()
    for i in range(size):
        cur_line = list()
        y = y_start + s * i
        for j in range(size):
            # print(i, j)
            x = x_start + s * j
            cur_prediction = predict_for([x, y, 1], dataset)
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

    show_pic(data_pic, name)


def test_on_blocks(file, name_parameter, parameter_value):
    file.write("Block testing for parameter %s = %.3f\n" % (name_parameter, parameter_value))
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
        find_a_b(train_dataset, test_dataset)
        ress = calc_accuracy(train_dataset, test_dataset)
        file.write("Accuracy for block %d = %.4f\n" % (i, ress))
        sum_accuracy += ress
    file.write("Average accuracy=%.4f\n\n" % (sum_accuracy / num_blocks))
    return sum_accuracy / num_blocks


def find_best_parameter(parameter_name, type_kernel_name):
    str = "research/" + type_kernel_name + "/research_result_for_parameter_" + parameter_name + ".txt"
    file = open(str, "w")
    best_accuracy = -1
    best_parameter = parameter_choice_dic[parameter_name][0]
    for i in parameter_choice_dic[parameter_name]:
        cur_accuracy = test_on_blocks(file, parameter_name, i)
        if cur_accuracy > best_accuracy:
            best_parameter = i
            best_accuracy = cur_accuracy
    file.write("Best parameter %s = %.3f\n" % (parameter_name, best_parameter))
    param_name_variable_dic[parameter_name](best_parameter)
    file.close()


def test_and_print(type_kernel_name):
    global cur_core_function
    cur_core_function = kernel_dict[type_kernel_name]
    for str in parameter_for_kernel_type_dic[type_kernel_name]:
        find_best_parameter(str, type_kernel_name)
    find_a_b(dataset, dataset)
    drow_dataset(dataset, type_kernel_name)


kernel_dict = {
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "gaussian": gaussian_kernel
}

parameter_choice_dic = {
    "C": [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    "P": [2, 3, 4, 5],
    "B": [1, 2, 3, 4, 5]
}
# ,  0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
parameter_for_kernel_type_dic = {
    "linear": ["C"],
    "polynomial": ["C", "P"],
    "gaussian": ["C", "B"]
}

# PARAMETERS
num_blocks = 5
tol = 0.0001
max_counter = 500
cur_core_function = linear_kernel

# HYPER PARAMETERS
C = parameter_choice_dic["C"][-1]
P = parameter_choice_dic["P"][0]
B = parameter_choice_dic["B"][-1]

# dataset_name = "geyser.csv"
dataset_name = "chips.csv"
dataset = read_and_prepare_dataset(dataset_name)
# dataset = read_and_prepare_dataset("my.csv")
# dataset = read_and_prepare_dataset("chips.csv")
# print("average acc", test_on_blocks())

a = 0
b = 0
y = list()
K = list()

# find_a_b(dataset)

# print(calc_accuracy(a, b, dataset, dataset, cur_core_function))
#
# drow_dataset(dataset, a, b)
test_and_print("linear")
test_and_print("polynomial")
test_and_print("gaussian")

# find_best_parameter("C")

# average_accurancy = test_on_blocks()
