from math import tanh
from math import cosh


def get_shape(values, index):
    return len(values[index]), len(values[index][0])


def add_matrix(y, x):
    m_values.append([[0 for j in range(x)] for k in range(y)])
    m_ders.append([[0 for j in range(x)] for k in range(y)])


def calc_rlu(x, a):
    if (x < 0):
        return a * x
    return x


def calc_der_rlu(x, a):
    if (x < 0):
        return a
    return 1


def der_tanh(x):
    return 1 / cosh(x) ** 2


def forward():
    for i in range(M + 1, N + 1):
        type = m_types[i]
        if (type == 'var'):
            print("HELP1")
            continue
        if (type == 'mul'):
            C = m_values[i]
            A = m_values[m_inf[i][0]]
            B = m_values[m_inf[i][1]]
            n = len(A)
            m = len(A[0])
            k = len(B[0])
            for x in range(n):
                for y in range(m):
                    for z in range(k):
                        C[x][z] += A[x][y] * B[y][z]
            m_values[i] = C
            continue
        if (type == 'sum'):
            indexes = m_inf[i]
            C = m_values[i]
            for ind in indexes:
                A = m_values[ind]
                for y in range(len(C)):
                    for x in range(len(C[0])):
                        C[y][x] += A[y][x]
            m_values[i] = C
            continue
        if (type == 'had'):
            indexes = m_inf[i]
            C = m_values[i]
            [y, x] = [len(C), len(C[0])]
            cur_sum = [[1 for j in range(x)] for k in range(y)]
            for ind in indexes:
                A = m_values[ind]
                for y in range(len(C)):
                    for x in range(len(C[0])):
                        cur_sum[y][x] *= A[y][x]
            for y in range(len(C)):
                for x in range(len(C[0])):
                    C[y][x] += cur_sum[y][x]
            m_values[i] = C
            continue
        if (type == 'rlu'):
            [a_1, num] = m_inf[i]
            a = 1 / a_1
            C = m_values[i]
            A = m_values[num]
            for y in range(len(C)):
                for x in range(len(C[0])):
                    C[y][x] += calc_rlu(A[y][x], a)
            m_values[i] = C
            continue
        if (type == 'tnh'):
            num = m_inf[i][0]
            C = m_values[i]
            A = m_values[num]
            for y in range(len(C)):
                for x in range(len(C[0])):
                    C[y][x] += tanh(A[y][x])
            m_values[i] = C
            continue


def backward():
    for ind in range(N, 0, -1):
        type = m_types[ind]
        if (type == 'var'):
            continue
        if (type == 'mul'):
            ind_A = m_inf[ind][0]
            ind_B = m_inf[ind][1]
            [yA, xA] = get_shape(m_values, ind_A)
            [yB, xB] = get_shape(m_values, ind_B)
            for i in range(yA):
                for k in range(xB):
                    for j in range(xA):
                        m_ders[ind_A][i][j] += m_ders[ind][i][k] * m_values[ind_B][j][k]
                        m_ders[ind_B][j][k] += m_ders[ind][i][k] * m_values[ind_A][i][j]
            continue
        if (type == 'had'):
            indexes = m_inf[ind]
            [y, x] = get_shape(m_values, ind)
            for (i_, i) in enumerate(indexes):
                for j in range(y):
                    for k in range(x):
                        multi = 1
                        for (t_, t) in enumerate(indexes):
                            if t_ != i_:
                                multi *= m_values[t][j][k]
                        m_ders[i][j][k] += m_ders[ind][j][k] * multi
            continue
        if (type == 'sum'):
            indexes = m_inf[ind]
            [y, x] = get_shape(m_values, ind)
            for t in indexes:
                for i in range(y):
                    for j in range(x):
                        m_ders[t][i][j] += m_ders[ind][i][j]
            continue
        if (type == 'rlu'):
            [a_1, num] = m_inf[ind]
            a = 1 / a_1
            [y, x] = get_shape(m_values, ind)
            for i in range(y):
                for j in range(x):
                    m_ders[num][i][j] += m_ders[ind][i][j] * calc_der_rlu(m_values[num][i][j], a)
            continue
        if (type == 'tnh'):
            [num] = m_inf[ind]
            [y, x] = get_shape(m_values, ind)
            for i in range(y):
                for j in range(x):
                    m_ders[num][i][j] += m_ders[ind][i][j] * der_tanh(m_values[num][i][j])
            continue


def show_matrix(matr):
    for row in matr:
        for val in row:
            print(val, end=' ')
        print()


[N, M, K] = [int(elt) for elt in (input()).split()]
m_values = [[]]
m_ders = [[]]
m_types = ["null"]
m_inf = [[]]
fail = False
for i in range(1, N + 1):
    pars = input().split()
    type = pars[0]
    m_inf.append([])
    m_types.append(pars[0])
    if (type == 'var'):
        add_matrix(int(pars[1]), int(pars[2]))
        continue
    if (type == 'mul'):
        m_inf[i].append(int(pars[1]))
        m_inf[i].append(int(pars[2]))
        num1 = int(pars[1])
        num2 = int(pars[2])
        [y1, x1] = get_shape(m_values, num1)
        [y2, x2] = get_shape(m_values, num2)
        add_matrix(y1, x2)
        continue
    if (type == 'sum'):
        indexes = [int(x) for x in pars[2:]]
        m_inf[i] = indexes
        [y, x] = get_shape(m_values, indexes[0])
        add_matrix(y, x)
        continue
    if (type == 'had'):
        indexes = [int(x) for x in pars[2:]]
        m_inf[i] = indexes
        [y, x] = get_shape(m_values, indexes[0])
        add_matrix(y, x)
        continue
    if (type == 'rlu'):
        [a_1, x] = [int(x) for x in pars[1:]]
        m_inf[i] = [a_1, x]
        [y, x] = get_shape(m_values, x)
        add_matrix(y, x)
        continue
    if (type == 'tnh'):
        x = int(pars[1])
        m_inf[i].append(x)
        [y, x] = get_shape(m_values, x)
        add_matrix(y, x)
        continue
    fail = True
    break
if (not fail):
    for i in range(1, 1 + M):
        [y, x] = get_shape(m_values, i)
        for k in range(y):
            values = [int(arg) for arg in input().split()]
            for j in range(x):
                m_values[i][k][j] += values[j]
    for i in range(N - K + 1, N + 1):
        [y, x] = get_shape(m_values, i)
        for k in range(y):
            values = [int(arg) for arg in input().split()]
            for j in range(x):
                m_ders[i][k][j] += values[j]

    forward()
    backward()

    for i in range(N - K + 1, N + 1):
        show_matrix(m_values[i])
    for i in range(1, 1 + M):
        show_matrix(m_ders[i])
else:
    print("oh no")
# for i in range(1, N+1):
#     print("i=", i)
#     show_matrix(m_values[i])
#     print("der")
#     show_matrix(m_ders[i])
#     print()
