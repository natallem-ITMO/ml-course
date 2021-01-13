from math import tanh
from math import sqrt

N = int(input())
x = list()
y = list()
for i in range(N):
    [xx, yy] = [int(elt) for elt in (input()).split()]
    x.append(xx)
    y.append(yy)
x_ = sum(x)/ len(x)
y_ = sum(y) / len(y)
t1 = sum([(xx-x_)*(yy - y_) for (xx, yy) in zip(x,y)])
tx = sum([(xx - x_)**2 for xx in x])
ty = sum([(yy - y_)**2 for yy in y])
if (ty*tx == 0):
    print(0)
else:
    print( t1 / sqrt(tx*ty))