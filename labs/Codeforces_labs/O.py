K = int(input())
N = int(input())
x_y_map = {}
for i in range(N):
    [x, y] = [int(elt) for elt in (input()).split()]
    if not (x in x_y_map):
        x_y_map[x] = {}
    if not (y in x_y_map[x]):
        x_y_map[x][y] = 0
    x_y_map[x][y] += 1
matoz = 0
for i in range(K+1):
    if not i in x_y_map:
        continue
    ys = x_y_map[i].keys()
    all = sum(x_y_map[i].values())
    mat_2 = 0
    mat = 0
    for y in ys:
        pos = x_y_map[i][y] / all
        mat_2 += y*y * pos
        mat += y * pos
    mm = mat_2 - mat**2
    matoz += mm * all/N
print(matoz)

from math import tanh
from math import sqrt