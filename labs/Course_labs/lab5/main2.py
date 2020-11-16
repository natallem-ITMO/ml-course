from sklearn import tree
import pandas as pd
from math import sqrt
import random
from matplotlib import pyplot as plt

# dataset = 0
max_number_of_dataset = 21

criterion_choices = ['gini', 'entropy']
splitter_choices = ['best', 'random']
criterion_choice = criterion_choices[0]
splitter_choice = splitter_choices[1]


def getFileName(n, isTrain=1):
    if isTrain:
        return "datasets/" + "{0:0>2}".format(n) + "_train.csv"
    else:
        return "datasets/" + "{0:0>2}".format(n) + "_test.csv"


def load_dataset(n, isTrain=1):
    global dataset
    str = getFileName(n, isTrain)
    return pd.read_csv(str)


def calc_accuracy(tree, test_dataset):
    true_predicted = 0
    for i in range(0, len(test_dataset.index)):
        if test_dataset.loc[i, test_dataset.columns == 'y'].values == tree.predict(
                [test_dataset.loc[i, test_dataset.columns != 'y']]):
            true_predicted += 1
    return true_predicted / len(test_dataset.index)


def prediction_by_tree_result(num, clfs, datasets, indexes):
    # cacl prediction on each tree
    predictions = {}
    for (index, clf) in enumerate(clfs):
        cur_row = [datasets.loc[num, i] for i in indexes[index][:-1]]
        prediction = clf.predict([cur_row])[0]
        if (prediction in predictions):
            predictions[prediction] += 1
        else:
            predictions[prediction] = 1
    prediction = max(predictions, key=predictions.get)
    real = datasets.loc[num, 'y']
    return prediction == real


def calc_forest_accuracy(clfs, datasets, indexes):
    true_predicted = 0
    for i in range(len(datasets.index)):
        if (i % 100 == 0):
            print("calc accuracy for ", i, "out of", len(datasets.index), " for now true_predicted", true_predicted)
        if (prediction_by_tree_result(i, clfs, datasets, indexes)):
            true_predicted += 1
    return true_predicted / len(datasets.index)


def find_forest(num, file):
    train_dataset = load_dataset(num)
    test_dataset = load_dataset(num, 0)
    trees_cls = list()
    trees_indexes = list()
    # size_of_training_subset = len(train_dataset.index)
    for i in range(trees_number_in_forest):
        column_names = train_dataset.columns.values.copy()
        random.shuffle(column_names[:-1])
        features_number = len(column_names) - 1
        features_number_sqrt = int(features_number ** (1/2))
        new_features_name = [i for i in column_names[0:features_number_sqrt]] + ['y']

        rows_indexes = [i for i in range(0,size_of_training_subset)]
        random.shuffle(rows_indexes)
        # print(rows_indexes)
        rows_indexes = rows_indexes[:size_of_training_subset]
        # print(rows_indexes)
        # print(train_dataset[[2,3,4],['x1','y']])
        new_train_dataset1 = (train_dataset.copy().loc[:, new_features_name]).copy()
        new_train_dataset2 = (new_train_dataset1.loc[rows_indexes, :]).copy()

        trees_indexes.append(new_features_name)
        clf = tree.DecisionTreeClassifier(criterion=criterion_choice, splitter=splitter_choice)
        clf = clf.fit(new_train_dataset2.loc[:, new_train_dataset2.columns != 'y'].copy(),
                      new_train_dataset2.loc[:, new_train_dataset2.columns == 'y'].copy())
        # print(clf.score(new_train_dataset2.loc[:, new_train_dataset2.columns != 'y'].copy(),
        #                 new_train_dataset2.loc[:, new_train_dataset2.columns == 'y'].copy()))
        # print(clf.score(train_dataset.loc[:, train_dataset.columns != 'y'].copy(),
        #                 train_dataset.loc[:, train_dataset.columns == 'y'].copy()))
        trees_cls.append(clf)
    test_accuracy = calc_forest_accuracy(trees_cls, test_dataset, trees_indexes)
    train_accuracy = calc_forest_accuracy(trees_cls, train_dataset, trees_indexes)
    # train_accuracy = 1
    file.write("For %d dataset train accuracy=%.4f and test accuracy=%.4f\n" % (num, train_accuracy, test_accuracy))
    print("For %d dataset train accuracy=%.4f and test accuracy=%.4f\n" % (num, train_accuracy, test_accuracy))


def find_forests_for_all_datasets():
    file = open("research/find_forests_for_all_datasets.txt", "w")
    ii = 1
    for i in range(ii, ii + 1):
        find_forest(i, file)
    file.close()


size_of_training_subset = 100
trees_number_in_forest = 900
find_forests_for_all_datasets()
