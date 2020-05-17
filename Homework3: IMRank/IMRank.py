import random
import queue
import numpy as np
from time import *


def load_file(file_name="DUNF with Weights.txt"):  # 读取文件，文件形式为三列，分别为父节点、子节点、权重，返回图的邻接矩阵
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        num = int(max(max([float(_) for _ in line.split()]) for line in lines))  # 获取节点的个数
        data = np.zeros((num+1, num+1), np.float32)
        for line in lines:
            data[int(line.split()[0])][int(line.split()[1])] = float(line.split()[2])
    return data


def ic_model(data, seed_list, total=100):  # 输入带权图、爆发种子结点、爆发次数，返回键为感染点、值为感染几率的字典
    active_dict = {}
    for break_num in range(total):
        active_set = np.zeros(len(data), np.int)  # 标记点的感染状态，为0表示未感染，为1表示感染
        child_queue = queue.Queue()  # 待考察点
        for _ in seed_list:
            child_queue.put(_)
        while not child_queue.empty():  # 当队列不为空时，即存在已感染的点的孩子结点尚未考察
            i = child_queue.get()  # 本次考察的点
            for j in range(len(data)):  # 对于该点的每个孩子
                if j != i and data[i][j] != 0:  # 不为自己，也不为零
                    if active_set[i] != 1 and data[i][j] > random.random():  # 在此次传染中，由未感染变为感染
                        if j not in active_dict:
                            active_dict[j] = 1
                        else:
                            active_dict[j] += 1
                        active_set[i] = 1  # 设置为被感染的状态
                        child_queue.put(j)  # 加入待考察队列
    for _ in active_dict:
        active_dict[_] = active_dict[_] / total
    return active_dict


def last_to_first(mi_rank, data):  # 输入数据和现有排名，计算下一次迭代产生的边界影响力
    mi_list = [1 for _ in range(len(mi_rank))]  # 存放每个结点对应的Mr(r)，表示该点的边界影响力
    for i in range(len(mi_rank)-1, -1, -1):  # 取出排名为i的结点
        for j in range(i):
            mi_list[mi_rank[j]] = mi_list[mi_rank[j]] + data[mi_rank[j]][mi_rank[i]] * mi_list[mi_rank[i]]
            mi_list[mi_rank[i]] = (1 - data[mi_rank[j]][mi_rank[i]]) * mi_list[mi_rank[i]]
    return mi_list


def greedy_mi_influence(order1, order2, k, data):  # 传入两次迭代产生的结果，返回模拟爆发之后的序列
    result = []
    i = 0
    j = 0
    while len(result) < k:
        if order1[i] in result:
            i = i + 1
            continue
        if order2[j] in result:
            j = j + 1
            continue
        if order1[i] == order2[j]:  # 如果两处排名相同，则说明排名没有波动，则加入序列
            result.append(order1[i])
            i = i + 1
            j = j + 1
        else:
            tmp1 = sum(ic_model(data, result+order1[i:i+1:1]).values())  # 计算order1[i]的边界影响力
            tmp2 = sum(ic_model(data, result+order2[j:j+1:1]).values())   # 计算order2[j]的边界影响力
            if tmp1 > tmp2:
                result.append(order1[i])
                i = i + 1
            else:
                result.append(order2[j])
                j = j + 1
    return result


def im_rank(data, init_rank, top_k=20, iter_num=40):  # 返回的边界影响力是全部点的影响力
    mi_rank_1 = [_ for _ in init_rank]  # 第零次迭代产生的排名
    iter_count = 0  # 迭代次数
    equal = False
    while not equal and iter_count <= iter_num:  # 终止迭代条件（满足其一）：1.迭代不导致结果变动  2.达到一定迭代次数
        tmp = last_to_first(mi_rank_1, data)
        mi_rank_2 = np.argsort(np.array(tmp))[::-1].tolist()
        equal = (np.array(mi_rank_1) == np.array(mi_rank_2)).all()  # 如果两次迭代得到的边界影响最大的k个点相同，则停止迭代
        mi_rank_1 = [_ for _ in mi_rank_2]
        iter_count = iter_count + 1
    return mi_rank_2[:top_k:]


def combine_rank(data, top_k=20, iter_num=30):  # 综合采用两种方法计算影响力排名
    init_rank_1 = [len(data) - i - 1 for i in range(len(data))]  # 生成的初始排名1
    init_rank_2 = [i for i in range(len(data))]  # 生成的初始排名2
    mi_rank_1 = im_rank(data, init_rank_1, top_k, iter_num)
    mi_rank_2 = im_rank(data, init_rank_2, top_k, iter_num)
    mi_order = greedy_mi_influence(mi_rank_1, mi_rank_2, top_k, data)  # 迭代可能未收敛，采用贪婪算法再次处理
    return mi_order


def max_influence(data, top_k=5, iter_num=30, break_num=10):  # 返回影响力最大的k个点，和这k个点在模拟爆发时的平均感染点数
    begin_time = time()  # 开始计算边界影响力的时间
    order = combine_rank(data, top_k, iter_num)
    end_mi_cal = time()  # 边界影响力计算完成的时间
    duration1 = end_mi_cal - begin_time
    break_result = ic_model(data, order, break_num)  # 传入用于爆发的种子时，应该从order列表中选取前k个，k的选取为计算边界影响最大的的个数
    duration2 = time() - end_mi_cal  # 计算模拟爆发所用时长
    print("计算影响力最大的%d个点用时：%fs\n计算这%d点模拟爆发用时： %fs\n" % (top_k, duration1, top_k, duration2))
    active_count = 0
    for _ in break_result:
        active_count += break_result[_]
    return order, break_result


if __name__ == "__main__":
    Data = load_file("DUNF with Weights.txt")
    seedSetList = []  # 存放每次计算得到影响力最大的k个点
    activeSetList = []  # 存放感染的
    activeCountList = []  # 存放平均感染的点的个数
    iterNum = 30  # 计算边界影响力时的迭代次数上界
    breakNum = 100  # 模拟爆发点的次数
    for topK in list(range(5, 25, 5)):
        seedSet, activeSet = max_influence(Data, topK, iterNum, breakNum)
        seedSetList.append(seedSet)
        activeSetList.append(activeSet)
        activeCountList.append(sum(activeSet.values()))
    print("TOP5结点影响力打分： %f\nTOP10结点影响力打分： %f\nTOP15结点影响力打分： %f\nTOP20结点影响力打分： %f\n" % (activeCountList[0], activeCountList[1], activeCountList[2], activeCountList[3]))

