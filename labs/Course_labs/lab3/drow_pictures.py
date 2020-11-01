import matplotlib.pyplot as plt
# import numpy as np

# dat = np.random.random(200).reshape(20,10) # создаём матрицу значений
from numpy import save


def get_bounds(dataset):
    min_width = dataset[0][0]
    max_width = dataset[0][0]
    min_height = dataset[0][1]
    max_height = dataset[0][1]
    for row in dataset:
        min_width = min(min_width, row[0])
        max_width = max(max_width, row[0])
        min_height = min(min_height, row[1])
        max_height = max(max_height, row[1])
    return [min_width, max_width, min_height, max_height]

def add_margin(min_width, max_width, min_height, max_height):
    margin = 0.2
    x_start = min_width - (max_width-min_width) * margin
    x_end = max_width + (max_width-min_width) * margin
    y_start = min_height - (max_height - min_height) * margin
    y_end = max_height + (max_height - min_height) * margin
    return [x_start, x_end, y_start, y_end]
def drow_dataset(dataset):
    [min_width, max_width, min_height, max_height] = get_bounds(dataset)
    [x_start, x_end, y_start, y_end] = add_margin(min_width, max_width, min_height, max_height)
    print(x_start, x_end, y_start, y_end)

dat = list()
max_range = 500
for i in range(1, max_range):
    dat.append([i] * int(max_range * 2 / 10) + [i - 100] + [i] * int(max_range * 5 / 10))

print(len(dat))
print(len(dat[0]))
fig = plt.figure(figsize=(30, 30))
pc = plt.pcolor(dat)  # метод псевдографики pcolor
# plt.colorbar(pc)
plt.title('Simple pcolor plot')

#
#
# fig = plt.figure()
# cf = plt.contourf(dat)
# plt.colorbar(cf)
# plt.title('Simple contourf plot')
#
# fig = plt.figure()
# cf = plt.matshow(dat)
# plt.colorbar(cf, shrink=0.7)
# plt.title('Simple matshow plot')

# save(name='pic_2_4', fmt='pdf')
# save(name='pic_2_4', fmt='png')

# plt.show()
plt.show()
