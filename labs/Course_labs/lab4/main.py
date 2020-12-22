import glob
from matplotlib import pyplot as plt
from math import log
from math import exp

n = 1
K = 2
# l_spam, l_legin
# fines = [0.1, 10000000000000000000000000000000000000000000000]
fines = [0.0001, 0.00000001]

intense = 0.00001

spmsg_count = 0
legit_count = 0
correct_k_fold_predicted = 0
all_k_fold_predictions = 0
all_predictions_ROC = list()
false_negatives = 0


def train_by_part(cur_part, class_in_input, all_grams, gram_in_class, N):
    cur_part_name = "part" + str(cur_part)
    all_files_pattern = "dataset/" + cur_part_name + "/*.txt"
    for file in glob.glob(all_files_pattern):
        N += 1
        cur_class = 1
        if 'spmsg' in str(file):
            cur_class = 0
        all_words_in_message = list()
        with open(file) as f:
            all_words_in_message += ["s" + elf for elf in f.readline().split()[1:]]
            f.readline()
            all_words_in_message += f.readline().split()

        all_grams_in_message = set(
            tuple(all_words_in_message[i:(i + n)]) for i in range(len(all_words_in_message) - n + 1))
        for gram in all_grams_in_message:
            all_grams.add(gram)
            if gram in gram_in_class[cur_class]:
                gram_in_class[cur_class][gram] += 1
            else:
                gram_in_class[cur_class][gram] = 1
        class_in_input[cur_class] += 1
    return N


def create_table(class_in_input, table, all_grams, gram_in_class):
    for i in range(0, K):
        table[i] = {}
        for gram in all_grams:
            t = 0
            if gram in gram_in_class[i]:
                t = gram_in_class[i][gram]
            table[i][gram] = (t + intense) / (class_in_input[i] + 2 * intense)


def test_by_part(cur_part, class_in_input, table, N, for_ROC=False):
    global correct_k_fold_predicted, all_k_fold_predictions, spmsg_count, legit_count, false_negatives
    cur_part_name = "part" + str(cur_part)
    all_files_pattern = "dataset/" + cur_part_name + "/*.txt"
    correct_predicted = 0
    all_predictions = 0
    for file in glob.glob(all_files_pattern):
        all_predictions += 1
        if 'spmsg' in str(file):
            actual_class = '0'
            spmsg_count += 1
        else:
            actual_class = '1'
            legit_count += 1

        all_words_in_message = list()
        with open(file) as f:
            all_words_in_message += ["s" + elf for elf in f.readline().split()[1:]]
            f.readline()
            all_words_in_message += f.readline().split()
        all_grams_in_message = set(
            tuple(all_words_in_message[i:(i + n)]) for i in range(len(all_words_in_message) - n + 1))
        cur_result = list()
        for c in range(0, K):
            cl = c
            if class_in_input[cl] != 0:
                start = log(class_in_input[cl] / (N) * fines[c])
                for gram_in_train in table[cl]:
                    if gram_in_train in all_grams_in_message:
                        start += log(table[cl][gram_in_train])
                    else:
                        start += log(1 - table[cl][gram_in_train])
            else:
                start = 0
            cur_result.append(start)
        max_log = max(cur_result)
        cur_sum = sum([exp(t - max_log) for t in cur_result if t != 0])
        result_possibilities = list()
        for (ind, res) in enumerate(cur_result):
            if res == 0:
                result_possibilities.append(0.0)
                print(0.0, "", sep=' ', end='', flush=True)
            else:
                result_possibilities.append(exp(cur_result[ind] - max_log) / cur_sum)
        predicted_class = '0'
        if for_ROC:
            all_predictions_ROC.append((result_possibilities[1], actual_class))
        if result_possibilities[1] > result_possibilities[0]:
            predicted_class = '1'
        if predicted_class == actual_class:
            correct_predicted += 1
        if not for_ROC:
            if predicted_class == '0' and actual_class == '1':
                print("holy crap", result_possibilities[1])
                false_negatives += 1
    accuracy = correct_predicted / all_predictions
    correct_k_fold_predicted += correct_predicted
    all_k_fold_predictions += all_predictions
    return accuracy


def k_fold_validation(for_ROC=False, show_current_accuracy=False):
    global correct_k_fold_predicted, all_k_fold_predictions
    num_of_folds = 10
    for test_num in range(1, num_of_folds + 1):
        class_in_input = {}
        N = 0
        gram_in_class = {}
        for i in range(0, K):
            gram_in_class[i] = {}
            class_in_input[i] = 0
        all_grams = set()
        table = {}
        for train_num in range(1, num_of_folds + 1):
            if train_num == test_num:
                continue
            N = train_by_part(train_num, class_in_input, all_grams, gram_in_class, N)
            create_table(class_in_input, table, all_grams, gram_in_class)
        training_accuracy = test_by_part(test_num, class_in_input, table, N, for_ROC)
        if show_current_accuracy:
            print("training accuracy", test_num, training_accuracy)
    return correct_k_fold_predicted / all_k_fold_predictions


def draw_ROC():
    k_fold_validation(True, True)
    print("Final accuracy", correct_k_fold_predicted / all_k_fold_predictions)
    x = [0.0]
    y = [0.0]
    all_predictions_ROC_sorted = sorted(all_predictions_ROC, key=lambda student: student[0], reverse=True)
    for tup in all_predictions_ROC_sorted:
        value = tup[1]
        if value == '1':
            x.append(x[-1])
            y.append(y[-1] + 1 / legit_count)
        else:
            x.append(x[-1] + 1 / spmsg_count)
            y.append(y[-1])

    plt.plot(x, y, '.-y', alpha=0.6, label="ROC for legit messages", lw=3)
    plt.legend()
    plt.show()


def draw_plot():
    global fines, correct_k_fold_predicted, all_k_fold_predictions, false_negatives
    fines = [0.1, 0.1]
    end_fine_value = 10000000000000000000000000000000000000000000000
    second = fines[0]
    x = list()
    y = list()
    while second <= end_fine_value:
        fines = [0.1, second]
        x.append(second)
        correct_k_fold_predicted = 0
        all_k_fold_predictions = 0
        false_negatives = 0
        y.append(k_fold_validation(False, False))
        second *= 10
        print("current fines", fines, y[-1])
    assert (false_negatives == 0)
    plt.plot(x, y, '.-r', alpha=0.6, label="accuracy", lw=3)
    plt.xscale("log")
    plt.xlabel("lambda legit")
    plt.ylabel("k-fold accuracy")
    plt.legend()
    plt.show()


draw_ROC()
draw_plot()
