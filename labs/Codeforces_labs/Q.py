from math import log
[K1, K2] = [int(elt) for elt in (input()).split()]
N = int(input())
x_y_map = {}
for i in range(N):
    [x, y] = [int(elt) for elt in (input()).split()]
    if not (x in x_y_map):
        x_y_map[x] = {}
    if not (y in x_y_map[x]):
        x_y_map[x][y] = 0
    x_y_map[x][y] += 1
summ = 0
for x in x_y_map.keys():
    p_x_i = sum(x_y_map[x].values())
    cur_sum = 0
    for y in x_y_map[x].keys():
        pp = x_y_map[x][y] / p_x_i
        cur_sum += pp * log(pp)
    summ += (p_x_i/N) * cur_sum
print(-summ)
