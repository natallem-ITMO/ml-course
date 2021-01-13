from math import tanh
from math import sqrt


def get_diffs(v):
    if (len(v) < 2) :
        return 0
    cur_sum = sum([v[i] - v[0] for i in range(len(v))])
    all_sum = cur_sum
    for i in range(1, len(v)):
        diff = v[i] - v[i - 1]
        cur_sum = cur_sum + diff * (2 * i - len(v))
        all_sum += cur_sum
    return all_sum


K = int(input())
N = int(input())
Klist = [[] for i in range(K + 1)]
all = list()
for i in range(N):
    [x, y] = [int(elt) for elt in (input()).split()]
    Klist[y].append(x)
    all.append(x)
for i in range(K + 1):
    Klist[i].sort()
all.sort()
summa = get_diffs(all)
summa_inner = 0
for i in range(K+1):
    summa_inner += get_diffs(Klist[i])
print(summa_inner)
print(summa - summa_inner)

# for i in range(1, K + 1):
#     print(i, Klist[i])
# print(all)
