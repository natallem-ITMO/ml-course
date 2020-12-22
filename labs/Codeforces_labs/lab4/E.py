from math import log
from math import exp

K = int(input())
fines = [int(elt) for elt in (input()).split()]
intense = int(input())
N = int(input())
word_in_class = {}
class_in_input = {}
all_words = set()
for i in range(N):
    he = [elt for elt in (input()).split()]
    cur_class = he[0]
    words = {elt for elt in he[2:]}
    for word in words:
        all_words.add(word)
        if cur_class in word_in_class:
            if word in word_in_class[cur_class]:
                word_in_class[cur_class][word] += 1
                continue
            else:
                word_in_class[cur_class][word] = 1
        else:
            word_in_class[cur_class] = {}
            word_in_class[cur_class][word] = 1
    if cur_class in class_in_input:
        class_in_input[cur_class] += 1
    else:
        class_in_input[cur_class] = 1

table = {}
for i in range(1, K+1):
    table[str(i)] = {}
    for word in all_words:
        t = 0
        if str(i) in word_in_class:
            if word in word_in_class[str(i)]:
                t = word_in_class[str(i)][word]
        else:
            class_in_input[str(i)] = 0
        table[str(i)][word] = (t + intense) / (class_in_input[str(i)] + 2 * intense)

M = int(input())
for i in range(M):
    he = [elt for elt in (input()).split()]
    words = set([elt for elt in he[1:]])
    down_part = 0
    cur_result = list()
    for c in range(1, K + 1):
        cl = str(c)
        if (class_in_input[cl] != 0):
            start = log(class_in_input[cl]/(N)*fines[c - 1])
            for word_of_train in table[cl]:
                if word_of_train in words:
                    start += log(table[cl][word_of_train])
                else:
                    start += log(1 - table[cl][word_of_train])
        else:
            start = 0
        cur_result.append(start)
    max_log = max(cur_result)
    cur_sum = sum([exp(t - max_log) for t in cur_result if t != 0])
    for (ind, res) in enumerate(cur_result):
        if (res == 0):
            print(0.0, "", sep=' ', end='', flush=True)
        else :
            print(exp(cur_result[ind] - max_log) / cur_sum, "", sep=' ', end='', flush=True)
    print()
