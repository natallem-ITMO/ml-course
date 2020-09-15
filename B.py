def read_table(n):
    rows = list()
    for i in range(n):
        innerList = [float(x) for x in (input()).split()]
        rows.append(innerList)
    return rows


def sum_in_row(table):
    sum_in_rows = [0.0] * len(table)
    for i in range(len(table)):
        for j in range(len(table)):
            sum_in_rows[i] += table[i][j]
    return sum_in_rows


def sum_in_colomn(table):
    sum_in_rows = [0.0] * len(table)
    for i in range(len(table)):
        for j in range(len(table)):
            sum_in_rows[j] += table[i][j]
    return sum_in_rows


def calc_diag_by_sum(table, sums):
    recall = list()
    for i in range(len(table)):
        if (sums[i] == 0):
            recall.append(0)
        else:
            recall.append(table[i][i] / sums[i])
    return recall


def calc_f1_score(prec, rec):
    f1 = list()
    for p, r in zip(prec, rec):
        if ((p + r) == 0):
            f1.append(0)
        else:
            f1.append((2 * p * r) / (p + r))
    return f1


def calc_weighted_func(class_count, f1_score, whole_sum):
    if (whole_sum == 0):
        return 0
    res = 0
    for p, r in zip(class_count, f1_score):
        res += p * r
    return res / whole_sum


# output_values = [row[-3] for row in rows]
n = int(input())
table = read_table(n)
sum_colomns = sum_in_colomn(table)
sum_rows = sum_in_row(table)
whole_sum = sum(sum_rows)
precision = calc_diag_by_sum(table, sum_rows)
recall = calc_diag_by_sum(table, sum_colomns)
f1_score = calc_f1_score(precision, recall)
weighted_f1 = calc_weighted_func(sum_rows, f1_score, whole_sum)
weighted_precision = calc_weighted_func(sum_rows, precision, whole_sum)
weighted_recall = calc_weighted_func(sum_rows, recall, whole_sum)
first_num = 0
if (weighted_precision + weighted_recall != 0):
    first_num = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
print(first_num)
print(weighted_f1)
