def containsX(x,y):
    if x in X:
        if y in X[x]:
            return True
    return False

[K1, K2] = [int(elt) for elt in (input()).split()]
N = int(input())
X = {}
allX = set()
allY = set()
sumY = {}
sumX = {}
for i in range(N):
    [x, y] = [int(elt) for elt in (input()).split()]
    if not y in sumY :
        sumY[y] = 0
    if not x in sumX :
        sumX[x] = 0
    if not x in X:
        X[x] = {}
    if not y in X[x]:
        X[x][y] = 0
    X[x][y] += 1
    sumY[y] +=1
    sumX[x] +=1
    allX.add(x)
    allY.add(y)
sum1 = sum(sumX.values())
sum2 = sum(sumY.values())
if (sum1 != sum2) :
    print("OH NO")
sum = 0
for x in X.keys():
    for y in X[x].keys():
        sum += X[x][y] ** 2 / (sumX[x] * sumY[y] / sum1)
sum -= sum1
# for x in allX:
#     for y in allY:
#         if (containsX(x,y)) :
#             O = X[x][y]
#         else:
#             O = 0
#         E = sumX[x] * sumY[y] / sum1
#         sum += (O-E) ** 2 / E

print(sum)
#dsf




