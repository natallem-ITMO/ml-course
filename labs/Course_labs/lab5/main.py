from sklearn import tree
import pandas as pd
from matplotlib import pyplot as plt
import random

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


def find_optimal_depth_for_dataset(i, file):
    file.write("Calculating for file = %d\n" % (i))
    print("Calculating for file = %d\n" % (i))

    train_dataset = load_dataset(i)
    test_dataset = load_dataset(i, 0)
    clf = tree.DecisionTreeClassifier(criterion=criterion_choice, splitter=splitter_choice)
    clf = clf.fit(train_dataset.loc[:, train_dataset.columns != 'y'],
                  train_dataset.loc[:, train_dataset.columns == 'y'])
    max_depth = clf.get_depth()
    optimum_depth = max_depth
    optimum_criterion = "-"
    optimum_splitter = "-"
    max_accuracy = -1

    file.write("Max depth = %d with accurancy = %.4f\n" % (max_depth, max_accuracy))
    print("Max depth = %d with accurancy = %.4f\n" % (max_depth, max_accuracy))
    for cur_criterion in criterion_choices:
        for cur_splitter in splitter_choices:
            file.write("cur_criterion = %s cur_splitter = %s\n" % (cur_criterion, cur_splitter))
            print("cur_criterion = %s cur_splitter = %s\n" % (cur_criterion, cur_splitter))
            for cur_depth in range(1, max_depth + 1):
                cur_clf = tree.DecisionTreeClassifier(criterion=cur_criterion, splitter=cur_splitter,
                                                      max_depth=cur_depth)
                cur_clf = cur_clf.fit(train_dataset.loc[:, train_dataset.columns != 'y'],
                                      train_dataset.loc[:, train_dataset.columns == 'y'])
                cur_accuracy = calc_accuracy(cur_clf, test_dataset)
                file.write("Depth = %d with accurancy = %.4f\n" % (cur_depth, cur_accuracy))
                print("Depth = %d with accurancy = %.4f\n" % (cur_depth, cur_accuracy))
                if cur_accuracy > max_accuracy:
                    optimum_depth = cur_depth
                    max_accuracy = cur_accuracy
                    optimum_criterion = cur_criterion
                    optimum_splitter = cur_splitter
    file.write("Optimum depth = %d with max accuracy = %.4f and criterion=%s splitter=%s\n\n" % (
        optimum_depth, max_accuracy, optimum_criterion, optimum_splitter))
    print("Optimum depth = %d with max accuracy = %.4f and criterion=%s splitter=%s\n\n" % (
        optimum_depth, max_accuracy, optimum_criterion, optimum_splitter))
    return optimum_depth


def find_optimal_depth_for_all_datasets():
    file = open("research/find_optimal_depth_for_all_datasets.txt", "w")
    min_opt_depth = 1000000000000000000000
    min_opt_depth_i = -1

    max_opt_depth = 0
    max_opt_depth_i = -1
    for i in range(1, 1 + 1):
        cur_opt_depth = find_optimal_depth_for_dataset(i, file)
        if (cur_opt_depth < min_opt_depth):
            min_opt_depth = cur_opt_depth
            min_opt_depth_i = i
        if (cur_opt_depth > max_opt_depth):
            max_opt_depth = cur_opt_depth
            max_opt_depth_i = i
    file.write("Min opt depth = %d for dataset %d, max opt depth = %d for dataset %d" % (
        min_opt_depth, min_opt_depth_i, max_opt_depth, max_opt_depth_i))
    file.close()


def draw_graphic():
    global criterion_choice, splitter_choice
    min_criterion_choice = criterion_choices[0]
    min_splitter_choice = splitter_choices[0]
    min_depth = 1
    min_dataset_num = 3
    min_train_dataset = load_dataset(min_dataset_num)
    min_test_dataset = load_dataset(min_dataset_num, 0)
    min_x = list()
    clf_min = tree.DecisionTreeClassifier(criterion=min_criterion_choice, splitter=min_splitter_choice)
    clf_min = clf_min.fit(min_train_dataset.loc[:, min_train_dataset.columns != 'y'],
                          min_train_dataset.loc[:, min_train_dataset.columns == 'y'])
    max_min_depth = clf_min.get_depth()

    max_criterion_choice = criterion_choices[1]
    max_splitter_choice = splitter_choices[0]
    max_depth = 11
    max_dataset_num = 21
    max_train_dataset = load_dataset(max_dataset_num)
    max_test_dataset = load_dataset(max_dataset_num, 0)
    max_x = list()
    clf_max = tree.DecisionTreeClassifier(criterion=max_criterion_choice, splitter=max_splitter_choice)
    clf_max = clf_max.fit(max_train_dataset.loc[:, max_train_dataset.columns != 'y'],
                          max_train_dataset.loc[:, max_train_dataset.columns == 'y'])
    max_max_depth = clf_max.get_depth()

    y = list()

    for depth in range(1, max(max_min_depth, max_max_depth) + 1):
        print(depth)
        y.append(depth)
        cur_min_clf = tree.DecisionTreeClassifier(criterion=min_criterion_choice, splitter=min_splitter_choice,
                                                  max_depth=depth)
        cur_min_clf = cur_min_clf.fit(min_train_dataset.loc[:, min_train_dataset.columns != 'y'],
                                      min_train_dataset.loc[:, min_train_dataset.columns == 'y'])
        cur_accuracy = calc_accuracy(cur_min_clf, min_test_dataset)
        min_x.append(cur_accuracy)

        cur_max_clf = tree.DecisionTreeClassifier(criterion=max_criterion_choice, splitter=max_splitter_choice,
                                                  max_depth=depth)
        cur_max_clf = cur_max_clf.fit(max_train_dataset.loc[:, max_train_dataset.columns != 'y'],
                                      max_train_dataset.loc[:, max_train_dataset.columns == 'y'])
        cur_accuracy = calc_accuracy(cur_max_clf, max_test_dataset)
        max_x.append(cur_accuracy)

    plt.plot(y, min_x, '.-y', alpha=0.6, label="min", lw=5)
    plt.plot(y, max_x, '.-r', alpha=0.6, label="max", lw=5)
    plt.show()


find_optimal_depth_for_all_datasets()
draw_graphic()


# Forest task

def prediction_by_forest(num, clfs, datasets, indexes):  # calc prediction on each tree
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
    return prediction == realG.cpp


def calc_forest_accuracy(clfs, datasets, indexes):
    true_predicted = 0
    for i in range(len(datasets.index)):
        if (i % 100 == 0):
            print("calc accuracy for", i, "out of", len(datasets.index), "true_predicted =", true_predicted)
        if (prediction_by_forest(i, clfs, datasets, indexes)):
            true_predicted += 1
    return true_predicted / len(datasets.index)


def find_forest(num, file):
    train_dataset = load_dataset(num)
    test_dataset = load_dataset(num, 0)
    trees_cls = list()
    trees_indexes = list()
    for i in range(trees_number_in_forest):
        column_names = train_dataset.columns.values.copy()
        random.shuffle(column_names[:-1])
        features_number = len(column_names) - 1
        features_number_sqrt = int(features_number ** (0.8))
        new_features_name = [i for i in column_names[0:features_number_sqrt]] + ['y']

        new_train_dataset1 = (train_dataset.copy().loc[:, new_features_name]).copy()
        new_train_dataset2 = new_train_dataset1.sample(n=len(new_train_dataset1.index), replace=True)

        trees_indexes.append(new_features_name)
        clf = tree.DecisionTreeClassifier(criterion=criterion_choice, splitter=splitter_choice)
        clf = clf.fit(new_train_dataset2.loc[:, new_train_dataset2.columns != 'y'].copy(),
                      new_train_dataset2.loc[:, new_train_dataset2.columns == 'y'].copy())
        trees_cls.append(clf)
    test_accuracy = calc_forest_accuracy(trees_cls, test_dataset, trees_indexes)
    train_accuracy = calc_forest_accuracy(trees_cls, train_dataset, trees_indexes)
    file.write("For %d dataset train accuracy=%.4f and test accuracy=%.4f\n" % (num, train_accuracy, test_accuracy))
    print("For %d dataset train accuracy=%.4f and test accuracy=%.4f\n" % (num, train_accuracy, test_accuracy))


def find_forests_for_all_datasets():
    file = open("research/find_forests_for_all_datasets.txt", "w")
    for i in range(1, max_number_of_dataset + 1):
        find_forest(i, file)
    file.close()


trees_number_in_forest = 100
find_forests_for_all_datasets()
