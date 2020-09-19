def destribute_input(classes_num):
    input_array = [int(x) for x in (input()).split()]
    classes = list()
    for i in range(classes_num):
        classes.append([])
    for i, j in enumerate(input_array) :
        classes[j-1].append(i+1)
    return classes

def to_parts(classes, part_count):
    start = list()
    for ls in classes:
        start += ls
    return chunkify(start, part_count)

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

[N,M,K] = [int(x) for x in (input()).split()]
class_index = destribute_input(M)
res = to_parts(class_index, K)
for arr  in res :
    print (len(arr), *arr)
