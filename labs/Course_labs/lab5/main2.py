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
splitter_choice = splitter_choices[0]


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


def prediction_by_tree_result(num, clfs, datasets):
    # cacl prediction on each tree
    predictions = {}
    for (index, clf) in enumerate(clfs):
        cur_row = datasets[index].loc[num, datasets[index].columns != 'y']
        prediction = clf.predict([datasets[index].loc[num, datasets[index].columns != 'y']])[0]
        if (prediction in predictions):
            predictions[prediction] += 1
        else:
            predictions[prediction] = 1
    prediction = max(predictions, key=predictions.get)
    real = datasets[0].loc[num, datasets[0].columns == 'y'][0]
    return prediction == real


def calc_forest_accuracy(clfs, datasets):
    true_predicted = 0
    for i in range(len(datasets[0].index)):
        if (prediction_by_tree_result(i, clfs, datasets)):
            true_predicted += 1
    return true_predicted / len(datasets[0].index)


def find_forest(num, file):
    train_dataset = load_dataset(num)
    test_dataset = load_dataset(num, 0)
    trees_train = list()
    trees_tests = list()
    trees_cls = list()
    for i in range(trees_number_in_forest):
        column_names = test_dataset.columns.values
        random.shuffle(column_names[:-1])
        features_number = len(column_names) - 1
        features_number_sqrt = int(sqrt(features_number))
        new_features_name = [i for i in column_names[0:features_number_sqrt]] + ["y"]
        new_train_dataset = train_dataset[new_features_name].copy()
        new_test_dataset = test_dataset[new_features_name].copy()
        trees_train.append(new_train_dataset)
        trees_tests.append(new_test_dataset)
        clf = tree.DecisionTreeClassifier(criterion=criterion_choice, splitter=splitter_choice)
        clf = clf.fit(new_train_dataset.loc[:, new_train_dataset.columns != 'y'],
                      new_train_dataset.loc[:, new_train_dataset.columns == 'y'])
        trees_cls.append(clf)
    train_accuracy = calc_forest_accuracy(trees_cls, trees_train)
    test_accuracy = calc_forest_accuracy(trees_cls, trees_tests)
    file.write("For %d dataset train accuracy=%.4f and test accuracy=%.4f\n" % (num, train_accuracy, test_accuracy))
    print("For %d dataset train accuracy=%.4f and test accuracy=%.4f\n" % (num, train_accuracy, test_accuracy))


def find_forests_for_all_datasets():
    file = open("research/find_forests_for_all_datasets.txt", "w")
    for i in range(1, max_number_of_dataset + 1):
        find_forest(i, file)
    file.close()


trees_number_in_forest = 30
find_forests_for_all_datasets()
