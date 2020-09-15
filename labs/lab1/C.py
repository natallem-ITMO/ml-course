from math import pi
from math import e
from math import sqrt
from math import cos


def uniform_K(u):
    return 0.5 if abs(u) < 1 else 0


def triangular_K(u):
    return (1 - abs(u)) if abs(u) < 1 else 0


def epanechnikov_K(u):
    return 0.75 * (1 - u * u) if abs(u) < 1 else 0


def quartic_K(u):
    return 15 / 16 * (1 - u * u) ** 2 if abs(u) < 1 else 0


def triweight_K(u):
    # 7 test
    return 35 / 32 * (1 - u ** 2) ** 3 if abs(u) < 1 else 0


def tricube_K(u):
    # 4 test
    return 70 / 81 * (1 - (abs(u)) ** 3) ** 3 if abs(u) < 1 else 0


def gaussian_K(u):
    return (1 / sqrt(2 * pi)) * e ** (-1 / 2 * u * u)


def cosine_K(u):
    # 10 TEST
    return pi / 4 * cos(pi / 2 * u) if abs(u) < 1 else 0


def logistic_K(u):
    return 1 / (e ** u + e ** (-u) + 2)


def sigmoid_K(u):
    return 2 / pi / (e ** u + e ** (-u))


def manhattan_distance(row1, row2):
    distance = 0.0
    for i, j in zip(row1, row2[:-1]):
        distance += abs(i - j)
    return distance


def euclidean_distance(row1, row2):
    distance = 0.0
    for i, j in zip(row1, row2[:-1]):
        distance += (i - j) ** 2
    return sqrt(distance)


def chebyshev_distance(row1, row2):
    #in 4 7 10 test
    distance = 0.0
    for i, j in zip(row1, row2[:-1]):
        distance = max(distance, abs(i - j))
    return distance


def read_ds():
    ds = list()
    n = int(input().split()[0])
    for i in range(n):
        ds.append([float(x) for x in (input()).split()])
    return ds


def sort_dtrain(train, test_row, func_distance):
    distances = list()
    for train_row in train:
        dist = func_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    return (distances)


metrics = {
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "chebyshev": chebyshev_distance
}

ks = {"uniform": uniform_K,
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


def get_zero_dist(sorted_dataset):
    zero_count = sum([1 for x in sorted_dataset if x[1] == 0])
    zero_sum = sum([x[0][-1] for x in sorted_dataset if x[1] == 0])
    if (zero_count != 0):
        return zero_sum / zero_count
    else:
        return sum([x[0][-1] for x in sorted_dataset]) / len(sorted_dataset)


def find_a(sorted_datatrain, window, k_function):
    sum_y = 0
    sum = 0
    if (window == 0):
        return get_zero_dist(sorted_dataset)
    for pair in sorted_datatrain:
        temp = k_function(pair[1] / window)
        sum_y += pair[0][-1] * temp
        sum += temp
    if (sum == 0):
        return get_zero_dist(sorted_dataset)
    return sum_y / sum


# output_values = [row[-3] for row in rows]
dtest = read_ds()
row = [float(x) for x in (input()).split()]
row.append(0.0)
metric = metrics[input()]
k_type = ks[input()]
window_type = input()
is_1_test = 1 if (k_type == uniform_K and window_type == "fixed" and row == [0.0, 0.0, 0.0]) else 0
sorted_dataset = sort_dtrain(dtest, row, metric)
window = 1
if (window_type == "variable"):
    window = sorted_dataset[int(input())][1]
    # todo [+1]
else:
    window = float(input())
# print("window", window)
res = find_a(sorted_dataset, window, k_type)
print(res)
