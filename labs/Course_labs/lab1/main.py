from math import pi
from math import e
from math import sqrt
from math import cos

import pandas as pd
from matplotlib import pyplot as plt


# DEBUG AND UTILE

def print_some_lines(dataset, message="", num_start=67, num_end=74):
    print(message)
    print(*dataset[num_start:num_end], sep='\n')


def update_max(dataset, metric_func_name, k_func_name, window_type_name, window, measure):
    global selected_metric, selected_K_func, selected_window_type, selected_window, selected_label_encoding, max_measure
    selected_metric = metric_func_name
    selected_K_func = k_func_name
    selected_window_type = window_type_name
    selected_window = window
    selected_label_encoding = dataset
    max_measure = measure


def print_selected_parameters():
    print("Result of measurement:\n"
          "preferred encoding way for label: %s\n"
          "preferred metric function: %s\n"
          "preferred K function: %s\n"
          "preferred window type: %s\n"
          "preferred window: %s\n"
          "score: %f" % (
              selected_label_encoding, selected_metric, selected_K_func, selected_window_type, selected_window,
              max_measure))


# DATASET PREPARATIONS
def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def to_oneHot(dataset, index, classes_num):
    num_to_vec = list()
    pattern = [0] * classes_num
    for i in range(classes_num):
        pattern[i] = 1
        num_to_vec.append(pattern.copy())
        pattern[i] = 0
    for i in range(len(dataset)):
        class_number = dataset[i][index]
        dataset[i] = dataset[i][:index] + num_to_vec[class_number] + dataset[i][index + 1:]
    return dataset


def read_and_prepare_dataset():
    dataset = pd.read_csv("prnn_fglass.csv")
    minmax = [[columnData.min(), columnData.max()] for (columnName, columnData) in dataset.iloc[:, :-1].iteritems()]
    normalized_dataset = normalize(dataset.values.tolist(), minmax)
    lookup = str_column_to_int(normalized_dataset, len(normalized_dataset[0]) - 1)
    onehot_dataset = to_oneHot(normalized_dataset.copy(), len(normalized_dataset[0]) - 1, len(lookup))
    return (len(dataset.index), len(lookup), normalized_dataset, onehot_dataset)


# PARAMETER FUNCTIONS

def uniform_K(u):
    return 0.5 if abs(u) < 1 else 0


def triangular_K(u):
    return (1 - abs(u)) if abs(u) < 1 else 0


def epanechnikov_K(u):
    return 0.75 * (1 - u * u) if abs(u) < 1 else 0


def quartic_K(u):
    return 15 / 16 * (1 - u * u) ** 2 if abs(u) < 1 else 0


def triweight_K(u):
    return 35 / 32 * (1 - u ** 2) ** 3 if abs(u) < 1 else 0


def tricube_K(u):
    return 70 / 81 * (1 - (abs(u)) ** 3) ** 3 if abs(u) < 1 else 0


def gaussian_K(u):
    return (1 / sqrt(2 * pi)) * e ** (-1 / 2 * u * u)


def cosine_K(u):
    return pi / 4 * cos(pi / 2 * u) if abs(u) < 1 else 0


def logistic_K(u):
    return 1 / (e ** u + e ** (-u) + 2)


def sigmoid_K(u):
    return 2 / pi / (e ** u + e ** (-u))


def manhattan_distance(row1, row2, excluded_columns=1):
    distance = 0.0
    for i, j in zip(row1, row2[:-excluded_columns]):
        distance += abs(i - j)
    return distance


def euclidean_distance(row1, row2, excluded_columns=1):
    distance = 0.0
    for i, j in zip(row1, row2[:-excluded_columns]):
        distance += (i - j) ** 2
    return sqrt(distance)


def chebyshev_distance(row1, row2, excluded_columns=1):
    distance = 0.0
    for i, j in zip(row1, row2[:-excluded_columns]):
        distance = max(distance, abs(i - j))
    return distance


def get_h_fixed(dataset_dist, window):
    return window


def get_h_variable(dataset_dist, window):
    return dataset_dist[window][1]


metric_dict = {
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "chebyshev": chebyshev_distance
}

k_dict = {"uniform": uniform_K,
          "triangular": triangular_K,
          "epanechnikov": epanechnikov_K,
          "quartic": quartic_K,
          "triweight": triweight_K,
          "tricube": tricube_K,
          "gaussian": gaussian_K,
          "cosine": cosine_K,
          "logistic": logistic_K,
          "sigmoid": sigmoid_K
          }


# COMPUTE PREDICTION
def sort_dataset_by(train, test_row, func_distance, without_colomns):
    distances = list()
    for train_row in train:
        dist = func_distance(test_row, train_row, without_colomns)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    return (distances)


def get_zero_dist(sorted_dataset, needToCheckZeroDist=True):
    if needToCheckZeroDist:
        k_for_all_dataset_train = [1 if x[1] == 0 else 0 for x in sorted_dataset]
        if sum(k_for_all_dataset_train) == 0:
            k_for_all_dataset_train = [1] * len(k_for_all_dataset_train)
    else:
        k_for_all_dataset_train = [1] * len(sorted_dataset)
    return k_for_all_dataset_train


def calculate_prediction_and_actual(dataset, row, metric_func, k_func, calculate_dominator_for_k_func, window,
                                    label_dimension):
    dataset_dist = sort_dataset_by(dataset, row, metric_func, label_dimension)
    h = calculate_dominator_for_k_func(dataset_dist, window)
    if h == 0:
        k_for_all_dataset_train = get_zero_dist(dataset_dist)
    else:
        k_for_all_dataset_train = [k_func(row_dist[1] / h) for row_dist in dataset_dist]
    sum_k = sum(k_for_all_dataset_train)
    if sum_k == 0:
        k_for_all_dataset_train = get_zero_dist(dataset_dist, False)
        sum_k = sum(k_for_all_dataset_train)
    prediction_vector = [0.0] * label_dimension
    for i in range(label_dimension):
        shift_in_dataset = -(i + 1)
        sum_y_i_mul_k = sum([y_j[0][shift_in_dataset] * k_j for k_j, y_j in zip(k_for_all_dataset_train, dataset_dist)])
        prediction_vector[shift_in_dataset] = sum_y_i_mul_k / sum_k
    if label_dimension == 1:
        predicted_value = int(prediction_vector[0])
        actual_value = row[-1]
    else:
        predicted_value = prediction_vector.index(max(prediction_vector)) + 1
        actual_vector = row[-label_dimension:]
        actual_value = actual_vector.index(max(actual_vector)) + 1
    return predicted_value, actual_value


# COMPUTE F MEASURE

def divide_diag_by_sums(table, sums):
    recall = list()
    for i in range(len(table)):
        if (table[i][i] == 0):
            recall.append(0)
        else:
            recall.append(table[i][i] / sums[i])
    return recall


def calc_f1_score(prec, rec):
    return [(0 if (p * r == 0) else (2 * p * r) / (p + r))
            for p, r in zip(prec, rec)]


def calc_weighted_func(class_count, f1_score, whole_sum):
    if (whole_sum == 0):
        return 0
    return sum([p * r for p, r in zip(class_count, f1_score)]) / whole_sum


def count_f_measure(table):
    sum_colomns = [sum(i) for i in zip(*table)]
    sum_rows = [sum(row) for row in table]
    whole_sum = sum(sum_rows)
    precision = divide_diag_by_sums(table, sum_rows)
    recall = divide_diag_by_sums(table, sum_colomns)
    f1_score = calc_f1_score(precision, recall)
    return calc_weighted_func(sum_rows, f1_score, whole_sum)


# MAIN FUNCTIONS

def leave_one_out_cross_validation_for(dataset, metric_func, k_func, count_dominator_for_k_func, window,
                                       label_dimension=1):
    print(window)
    binary_classification_matrix = [[0 for x in range(class_count)] for y in range(class_count)]
    for i in range(len(dataset)):
        dataset_train = [row for (j, row) in enumerate(dataset) if (j != i)]
        test_row = dataset[i]
        [prediction, actual] = calculate_prediction_and_actual(dataset_train, test_row, metric_func, k_func,
                                                               count_dominator_for_k_func, window, label_dimension)
        binary_classification_matrix[prediction - 1][actual - 1] += 1
    return count_f_measure(binary_classification_matrix)


def search_parameters():
    for (metric_func_name, metric_func) in metric_dict.items():
        for (k_func_name, k_func) in k_dict.items():
            for (window_type_name, window_type_inf) in window_type_dict.items():
                for window in window_type_inf[0]:
                    for dataset_name, encoding_type_inf in encoding_type_dict.items():
                        score = leave_one_out_cross_validation_for(encoding_type_inf[0], metric_func, k_func,
                                                                   window_type_inf[1],
                                                                   window, encoding_type_inf[1])
                        print(
                            # "dataset_decodin_methond %s metric_func %s k_func %s window_type_name %s window %s with score %f" % (
                            #     dataset_name, metric_func_name, k_func_name, window_type_name, window, score))
                            "%s  %s  %s  %s window=%s score=%f" % (
                                dataset_name, metric_func_name, k_func_name, window_type_name, window, score))
                        if (score > max_measure):
                            update_max(dataset_name, metric_func_name, k_func_name, window_type_name, window, score)


def draw_plot_for(window_type, start_range, end_range, factor):
    x = [i * factor for i in range(start_range, end_range)]
    y = [
        leave_one_out_cross_validation_for(encoding_type_dict[selected_label_encoding][0],
                                           metric_dict[selected_metric], k_dict[selected_K_func],
                                           window_type_dict[window_type][1], window,
                                           encoding_type_dict[selected_label_encoding][1])
        for window in x]
    plt.plot(x, y, '.-y', alpha=0.6, label="f1-measure", lw=5)
    plt.show()


# PROGRAM

[rows_num, class_count, labelEncoding_dataset, oneHot_dataset] = read_and_prepare_dataset()

start_fixed_range = 0
end_fixed_range = 40
start_variable_range = 0
end_variable_range = rows_num - 2

window_type_dict = {
    "fixed": ([0.5 * i for i in range(start_fixed_range, end_fixed_range) if i % 10 == 0], get_h_fixed)
    , "variable": ([i for i in range(start_variable_range, end_variable_range) if (i % 5 == 0)], get_h_variable)
}
encoding_type_dict = {
    "OneHot": [oneHot_dataset, class_count]
    , "LabelEncoding": [labelEncoding_dataset, 1]
}

selected_label_encoding = "OneHot"
selected_metric = "manhattan"
selected_K_func = "triweight"
selected_window_type = "variable"
selected_window = 10
max_measure = 0

# search_parameters()
print_selected_parameters()

# drow plot for variable window
draw_plot_for("variable", start_variable_range, end_variable_range, 1)
# drow plot for fixed window
draw_plot_for("fixed", start_fixed_range, end_fixed_range, 0.2)
