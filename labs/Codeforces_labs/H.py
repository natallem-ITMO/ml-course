def build_SKNF(result):
    disjunctions = list()
    for vec in result:
        if vec[-1] == 0:
            disjunctions.append(vec[:-1])
    for i, disj in enumerate(disjunctions):
        disj = [1 if x==0 else 0 for x in disj]
        pos = sum(disj)
        neg = len(disj) - pos
        new_list = list()
        for val in disj:
            if (val == 1):
                new_list.append(neg)
            else:
                new_list.append(-1)
        new_list.append(neg - 1 / 2)
        disjunctions[i] = new_list

    conjunctions = [1 for i in range(len(disjunctions))] + [-(len(disjunctions) - 1 / 2)]
    disjunctions.append(conjunctions)
    return disjunctions


def build_SDNF(result):
    conjunctions = list()
    for vec in result:
        if vec[-1] == 1:
            conjunctions.append(vec[:-1])
    for i, conj in enumerate(conjunctions):
        pos = sum(conj)
        neg = len(conj) - pos
        new_list = list()
        for val in conj:
            if (val == 1):
                new_list.append(1)
            else:
                new_list.append(-pos-1)
        new_list.append(-pos + 1 / 2)
        conjunctions[i] = new_list

    disjunctions = [1 for i in range(len(conjunctions))] + [-1 / 2]
    conjunctions.append(disjunctions)
    return (conjunctions)


M = int(input())
# M = 4
result = [[0], [1]]
for i in range(2, M + 1):
    result_2 = list()
    for i in result:
        result_2.append(i.copy())
    # result_2 = copy.deepcopy(result)
    for j in range(len(result)):
        result_2[j].append(1)
        result[j].append(0)
    result = result + result_2

zeros = 0
ones = 0
for i in range(len(result)):
    t = int(input())
    result[i].append(t)
    if (t == 1):
        ones += 1
    else:
        zeros += 1


def print_res(res):
    for i in res:
        for t in i:
            print(t, end=' ')
        print()


def only_zeros():
    # print("hello")
    print("1\n1")
    res = [[0 for i in range(M)] + [-1]]
    print_res(res)


def only_ones():
    # print("hello")

    print("1\n1")
    res = [[0 for i in range(M)] + [1]]
    print_res(res)


if (ones == 0):
    only_zeros()

if (zeros == 0):
    only_ones()
res = list()
if (ones != 0 and zeros != 0):
    if (zeros >= ones):
        res = build_SDNF(result)
    else:
        res = build_SKNF(result)
    print(2)
    print(len(res) - 1, 1)
    print_res(res)


# def mul(vec1, vec2):
#     return sum([f * j for f, j in zip(vec1, vec2)])
#
# def porog(mull):
#     if (mull == 0):
#         print("OH NO")
#         return -1
#     if mull > 0:
#         return 1
#     else:
#         return 0
#
# for input_test in result:
#     first_results = list()
#     cur_input = input_test[:-1] + [1]
#     for cur_nerv in res[:-1]:
#         if (porog( mul(cur_input, cur_nerv)) == -1):
#             print("df")
#         first_results.append(porog( mul(cur_input, cur_nerv)))
#     first_results.append(1)
#     actual = porog(mul(first_results, res[-1]))
#     expected = input_test[-1]
#     if (expected != actual):
#         print("differs")

# fines = [int(elt) for elt in (input()).split()]
# int main(){
#     cin >> M;
# vector<vector<int>> result = {{0},{1}};
# for (int i = 2; i <= M; i++){
#     vector<vector<int>> second_part = result;
# for (int j = 0; j < result.size(); j++){
#     second_part[j].push_back(1);
# result[j].push_back(0);
# }
# for (auto & vec : second_part){
#     result.push_back(vec);
# }
# }
# int zeros = 0;
# int ones = 0;
# for (int i =0; i < result.size(); i++){
# int t;
# cin >> t;
#
# result[i].push_back(t);
# if (t == 0)
# ++zeros;
# else
# ++ones;
# }
# if (zeros > ones) {
# build_SDNF(result);
# }
#
# for (auto & vec : result){
# for (auto num : vec){
#     cout << num << " ";
# }
# cout << "\n";
# }
#
#
# std::cout << "hell";
# }
#
