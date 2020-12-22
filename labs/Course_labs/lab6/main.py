from sklearn import tree
import pandas as pd
from math import log, exp, floor
from matplotlib import pyplot as plt

max_depth = 2
tracked_steps = [1, 2, 3, 5, 8, 13, 21, 34, 55]

criterion_choices = ['gini', 'entropy']
splitter_choices = ['best', 'random']
criterion_choice = criterion_choices[0]
splitter_choice = splitter_choices[0]
dataset_names = ['chips', 'geyser']
max_depth_dataset = [2, 3]
dataset_index = 0


def load_dataset():
    str = "datasets/" + dataset_names[dataset_index] + ".csv"
    return pd.read_csv(str)


def predict(trees, alphas, vec, max_bound):
    sample_sum = 0
    for (alpha, cur_tree) in zip(alphas[:max_bound], trees[:max_bound]):
        prediction = 1 if cur_tree.predict([vec]) == ['P'] else -1
        sample_sum += alpha * prediction
    pred = 1 if sample_sum > 0 else -1
    return pred


def calc_accuracy(trees, alphas, dataset, max_bound=-1):
    true_predicted = 0
    if max_bound == -1:
        max_bound = len(trees)
    for i in range(len(dataset.index)):
        pred = ['P'] if predict(trees, alphas, dataset.loc[i, dataset.columns != 'class'], max_bound) == 1 else ['N']
        if dataset.loc[i, dataset.columns == 'class'].values == pred:
            true_predicted += 1
    print("calculating accuracy with bound",  max_bound, true_predicted / len(dataset.index))
    return true_predicted / len(dataset.index)


def calc_weighted_error(prediction_result_success, weights):
    return sum([weights[index] for (index, res) in enumerate(prediction_result_success) if res == -1])


def calc_prediction_result_success(tree, dataset):
    result = list()
    for i in range(0, len(dataset.index)):
        if dataset.loc[i, dataset.columns == 'class'].values \
                != tree.predict([dataset.loc[i, dataset.columns != 'class']]):
            result.append(-1)
        else:
            result.append(1)
    return result


def adaboost():
    global max_depth, dataset
    dataset = load_dataset()
    max_steps = tracked_steps[-1]
    l = len(dataset.index)
    w = [1 / l for i in range(l)]
    trees = list()
    alphas = list()
    for j in range(max_steps):
        trees.append(tree.DecisionTreeClassifier(criterion=criterion_choice, splitter=splitter_choice,
                                                 max_depth=max_depth_dataset[dataset_index]))
        trees[-1].fit(dataset.loc[:, dataset.columns != 'class'],
                      dataset.loc[:, dataset.columns == 'class'], sample_weight=w)
        prediction_result_success = calc_prediction_result_success(trees[-1], dataset)
        N = calc_weighted_error(prediction_result_success, w)
        alphas.append(1 / 2 * log((1 - N) / N))
        sum_w = 0
        for k in range(l):
            w[k] = (w[k] * exp(-alphas[-1] * prediction_result_success[k]))
            sum_w += w[k]
        w = [t / sum_w for t in w]
        print("adaboost on step", j+1, "with accuracy", sum([1 for i in prediction_result_success if i == 1]) / len(dataset.index) )
    return (trees, alphas, dataset)


def draw_plot(trees, alphas, dataset):
    x = [i for i in range(1, tracked_steps[-1] + 1)]
    y = [calc_accuracy(trees, alphas, dataset, i) for i in x]
    plt.plot(x, y, '.-y', alpha=0.6, label="accuracy", lw=3)
    plt.xscale("linear")
    plt.xlabel("adaboost step")
    plt.ylabel("accuracy")
    plt.legend()
    names = ['plot', 'for', 'dataset', dataset_names[dataset_index]]
    strr = 'pictures/plots/' + '_'.join(names) + '.png'
    plt.savefig(strr, bbox_inches='tight')
    plt.close()
    print("saved", strr)


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
    margin = 0.05
    x_start = min_width - (max_width - min_width) * margin
    x_end = max_width + (max_width - min_width) * margin
    y_start = min_height - (max_height - min_height) * margin
    y_end = max_height + (max_height - min_height) * margin
    return [x_start, x_end, y_start, y_end]


def save_pic(data_pic, max_boost_step):
    fig = plt.figure(figsize=(30, 30))
    pc = plt.pcolor(data_pic)
    names = ['graphic', 'for', 'dataset', dataset_names[dataset_index], 'adaboost', 'steps', str(max_boost_step)]
    fig.suptitle(' '.join(names), fontsize=100)
    path_to_save = 'pictures/' + dataset_names[dataset_index] + '/' + '_'.join(names) + '.png'
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()
    print("saved", path_to_save)


def draw_dataset(dataset, trees, alphas, max_boost_step):
    dataset = dataset.values.tolist()

    mark_class_P = 1000
    mark_class_N = -1000
    mark_all_P = 400
    mark_all_N = -400

    [min_width, max_width, min_height, max_height] = get_bounds(dataset)
    [x_start, x_end, y_start, y_end] = add_margin(min_width, max_width, min_height, max_height)

    pixels = 200
    s_x = (x_end - x_start) / pixels
    s_y = (y_end - y_start) / pixels
    data_pic = list()
    for i in range(pixels):
        cur_line = list()
        y = y_start + s_y * i
        for j in range(pixels):
            x = x_start + s_x * j
            cur_prediction = predict(trees, alphas, [x, y], max_boost_step)
            if cur_prediction == 1:
                cur_line.append(mark_all_P)
            if cur_prediction == -1:
                cur_line.append(mark_all_N)
        data_pic.append(cur_line)
    for row in dataset:
        x_row = row[0]
        y_row = row[1]
        k_x = floor((x_row - x_start) / s_x)
        k_y = floor((y_row - y_start) / s_y)
        cur_color = mark_class_N if row[2] == 'N' else mark_class_P
        data_pic[k_y][k_x] = cur_color

    save_pic(data_pic, max_boost_step)


for i in range(0, len(dataset_names)):
    dataset_index = i
    [trees, alphas, dataset] = adaboost()
    draw_plot(trees, alphas, dataset)
    for j in range(len(trees)):
        if j+1 in tracked_steps:
            draw_dataset(dataset, trees, alphas, j+1)
 