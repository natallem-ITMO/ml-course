from math import tanh
from math import sqrt

N = int(input())
x = list()
y = list()
x_pairs = list()
y_pairs = list()
for i in range(N):
    [xx, yy] = [int(elt) for elt in (input()).split()]
    x.append(xx)
    y.append(yy)
    x_pairs.append((i,xx))
    y_pairs.append((i,yy))

x_pairs.sort(key=lambda tup: tup[1])
y_pairs.sort(key=lambda tup: tup[1])
x_rang = [0 for x in range(N)]
for (rang,(i,val)) in enumerate(x_pairs):
    x_rang[i] = rang+1
y_rang = [0 for y in range(N)]
for (rang,(i,val)) in enumerate(y_pairs):
    y_rang[i] = rang+1
# print(x_pairs)
# print(y_pairs)
# print(x_rang)
# print(y_rang)
n = N
summa = sum([(ri-si)**2 for (ri,si) in zip(x_rang, y_rang)])
if ((n-1)*n * (n+1)) < 0 :
    result = 1
else:
    result = 1- 6 / ((n-1)*n * (n+1)) *summa
print(result)
