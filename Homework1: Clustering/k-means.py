import numpy as np
import math
import random

"""
K-means算法：
    输入：输入各个点的坐标，坐标用n维向量表示
    输出：对于每个点，计算其聚类标签，输出NMI
    1.随机选择K各点，编号为1-K，作为最初始聚类中心，将各个点的位置向量存放于k_list列表中，
    对于聚类中心i，维护一个聚类列表division[i]，存放属于该聚类中心的点的编号。
    对于训练样本的一个点，保留其位置、聚类标签、实际标签（用于计算NMI值）、编号。
    2.对于每一个点，计算到每一个聚类中心的距离，将其标签设为距离最小的聚类中心在k_list中的编号，
    将属于聚类中心i的数据点的编号，加入对应的聚类列表division[i]
    3.对于聚类列表division[i]，加权平均、计算新的聚类中心，并用新的聚类中心的坐标，更新k_list，重新计算聚类列表
    4.计算NMI值，并输出
    5.重复2-4，直至一定次数
"""


class Point:
    """
    存储每一个数据点，属性包括：位置position(n维列表)；聚类产生的标签label；
    实际聚类标签answer；用来进行编号的order
    """
    def __init__(self, position, answer, order, label="null"):
        self.position = position  # 用ndarray数组存储
        self.answer = answer
        self.order = order
        self.label = label

    def show_point(self):  # 打印输出
        print("position: ", self.position, "\nlabel: ", self.label, "\norder:", self.order, "\nanswer: ", self.answer)


def load_file():
    # 读取文件，数据存放在二维ndarray数组data
    file_name = "./breast.txt"
    with open(file_name, 'r') as fp:
        fp = open(file_name, 'r')
        lines = fp.readlines()
        data = np.array([[float(_) for _ in line.split()] for line in lines])
    # 将数据赋值给以point为元素的列表
    seq = 0
    for _ in data:  # 数据的最后一列为实际的分类标签，用于计算准确率
        pos = _[:-1:1]
        ans = int(_[-1] / 2) - 1  # 将用于验证的实际分类标签从2.0、4.0转化为0、1
        point_list.append(Point(pos, ans, seq))
        seq = seq + 1


def calc_label():
    # 根据聚类中心计算点的标签
    # print("division：", division)
    for point in point_list:
        distance = [np.sqrt(np.sum(np.square(point.position - center))) for center in k_list]
        # print("distance: ", distance)
        point.label = distance.index(min(distance))  # 将标签设为距离最近的聚类中心在k_list中的编号
        division[point.label].append(point.order)  # 对每个聚类中心，维护一个属于该聚类中心的点的集合（用order表示）


# 根据各个点的标签，更新聚类中心
def update_cluster():
    tmp = [0 for _ in range(len(point_list[0].position))]  # 用于保存新的计算所得的聚类中心
    for center_order in range(len(k_list)):  # 对于每个聚类中心
        for p in division[center_order]:  # 对于属于该聚类中心的各个点
            tmp = [tmp[i] + point_list[p].position[i] for i in range(len(point_list[p].position))]  # 计算各个维度累加距离
        tmp = [tmp[i] / len(division[center_order]) for i in range(len(tmp))]  # 计算加权中心
        k_list[center_order] = [tmp[i] for i in range(len(tmp))]  # 用计算所得的加权中心，更新k_list列表中聚类中心的位置
        # print("k_list[center_order]: ", k_list[center_order])  # 用于打印聚类中心点的坐标


def verify():  # 计算NMI，即归一化互信息，所用变量为存储在data_list中已经分类好的点
    """
    参考链接1：https://www.jianshu.com/p/43318a3dc715
    参考链接2：https://blog.csdn.net/hang916/article/details/88783931
    :return: 打印输出，并返回NMI值
    """
    p_grp_gnd = [[0 for i in range(k)] for j in range(ans_count)]  # 联合条件概率分布：grp表示聚类后的group，gnd表示ground truth
    p_grp = [0 for i in range(k)]  # grp表示聚类后的group边界分布
    p_gnd = [0 for i in range(ans_count)]  # gnd表示ground truth边界分布
    for p in point_list:
        p_grp[p.label] += 1  # 统计聚类产生的标签
        p_gnd[p.answer] += 1  # 统计实际分类的标签
        p_grp_gnd[p.label][p.answer] += 1  # 用于计算联合概率分布
    p_grp = [i / len(point_list) for i in p_grp]  # 计算聚类为group的边界概率分布
    p_gnd = [i / len(point_list) for i in p_gnd]  # 计算实际ground truth的边界概率分布
    p_grp_gnd = [[i / len(point_list) for i in t] for t in p_grp_gnd]  # 计算联合概率分布
    h_grp = -sum([i*math.log(i, 2) for i in p_grp])  # 计算聚类结果的信息熵
    h_gnd = -sum([i*math.log(i, 2) for i in p_gnd])  # 计算实际结果的信息熵
    # h_grp_gnd = sum([p_grp[i] * (math.log(p_gnd[i], 2) - math.log(p_grp[i], 2)) for i in range(len(p_grp))])  # 计算相对熵
    tmp = sum([sum([p_grp_gnd[i][j]*(math.log(p_grp_gnd[i][j], 2) - math.log(p_grp[i]*p_gnd[j], 2)) for i in range(2)]) for j in range(2)])
    nmi = 2 * tmp / (h_grp + h_gnd)
    print("NMI: %.4f" % nmi)
    return nmi


if __name__ == "__main__":
    # 初始化数据
    point_list = []  # 以Point为元素的列表，用于存储输出点的信息
    k = 2  # K-means参数
    ans_count = 2  # 实际聚类的标签种类
    load_file()  # 加载数据
    # 初始化k_list和division
    k_list = np.array([point_list[random.randint(0, len(point_list) - 1)].position for i in range(k)])  # 初始随机生成的聚类中心的坐标
    division = [[] for _ in range(len(k_list))]  # 全局变量，用于保存属于聚类中心的点的编号

    n = 1  # 迭代计算
    while n < 20:
        print("第 ", n, " 次迭代：")
        calc_label()
        update_cluster()
        verify()
        n = n + 1
