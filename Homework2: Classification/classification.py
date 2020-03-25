import numpy as np
import math


def load_file(file_name):  # 读取文件，返回保存信息的二维list，和保存标签的一维list
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        data = np.array([[float(_) for _ in line.split()[:len(line.split())-1]] for line in lines])  # 读取数据
        label = np.array([int(float(line.split()[-1]))-3 for line in lines])  # 读取标签，并转化为-1，1两个类别，得到一维list
    return data, label


def calc_cov(data):  # 传入list，返回协方差矩阵
    mean_list = np.mean(data, axis=0)  # 计算每列平均值
    data_copy = np.array([[line[i]-mean_list[i] for i in range(len(mean_list))] for line in data])  # 使均值为零
    covariance = np.dot(data_copy.T, data_copy) / (len(data_copy)-1)  # 计算协方差矩阵
    return covariance


def top_ratio(mat, ratio=0.95):  # 从矩阵中选出最大的k个特征值对应的特征向量，并将这些特征向量组成矩阵，其中前k个特征值占所有特征值之和的比例为ratio
    e_vals, e_vecs = np.linalg.eig(mat)  # 列向量才是特征向量
    sort_indices = np.argsort(e_vals)  # 返回一个同等维度的list，其中中存放可以将传入列表按从小到大排列时的下标
    ratio_list = np.cumsum(e_vals, axis=0) / np.cumsum(e_vals, axis=0)[-1]
    k = np.where(ratio_list == [e_val for e_val in ratio_list if e_val > ratio][0])[0][0]  # 得到k值
    # print("保留前  %d  个最大特征值对应的特征向量" % (k+1))
    return e_vals[sort_indices[-1:-k-2:-1]], e_vecs[:, sort_indices[-1:-k-2:-1]]


def dim_reduction(data, e_vecs):
    e_vecs_norm = [[t / np.sum(np.square(line)) for t in line] for line in e_vecs.T]  # 标准化
    data = [np.dot(e_vecs_norm, data[i]) for i in range(len(data))]  # 将数据样本进行投影
    return data


def separate_by_class(data, label):  # 按标签进行分类
    separated_class = {}
    for i in range(len(data)):
        if label[i] not in separated_class:
            separated_class[label[i]] = []
        separated_class[label[i]].append(data[i])
    return separated_class


# 提取特征属性，对属于同一个标签的，计算均值和方差
def cal_sta(data):  # 此处传入的data，为训练集中，属于同一个标签的数据
    # 将传入的list转化为ndarray数组
    mean_list = np.mean(np.array(data), axis=0)
    std_dev_list = np.sqrt(np.sum([np.square([data[line][i] - mean_list[i] for i in range(len(mean_list))]) for line in range(len(data))], axis=0) / (len(data) - 1))
    # print("mean_list: ", mean_list)
    # print("var_list: ", var_list)
    return mean_list, std_dev_list


def cal_gauss_prob(x, mean, stdev):  # 计算高斯概率密度
    exponent = math.exp(-math.pow(x-mean, 2) / (2 * math.pow(stdev, 2)))
    return exponent / math.sqrt(2 * math.pi * stdev)


def summarize(separated_class):  # 返回一个字典，字典的key为标签，summary包含每个标签下的每个属性的均值和标准差
    summary = {}
    for key in separated_class:
        summary[key] = []
        mean_list, std_dev_list = cal_sta(separated_class[key])
        summary[key].append({"mean_list": mean_list})
        summary[key].append({"standard_deviation_list": std_dev_list})
    # print(summary)
    return summary


def cal_class_label(summary, input_vector):  # 对每个属性的每个取值，计算概率
    probability = {}
    key_list = list(summary.keys())
    for key in key_list:
        probability[key] = 1
        for i in range(len(input_vector)):
            probability[key] *= cal_gauss_prob(input_vector[i], summary[key][0]["mean_list"][i], summary[key][1]["standard_deviation_list"][i])
    label = max(probability, key=probability.get)
    return label


def get_nmi(test_label, calc_label):  # 传入两个标签列表，根据这两个标签列表来计算互信息
    test_label_rm_duplicate = list(set(test_label))  # 测试训练集的标签去除重复
    calc_label_rm_duplicate = list(set(calc_label))  # 预测得到的表标签去除重复
    dic_test = dict(zip(test_label_rm_duplicate, range(len(test_label_rm_duplicate))))  # 生成字典，主要是想对标签进行编号
    dic_calc = dict(zip(calc_label_rm_duplicate, range(len(calc_label_rm_duplicate))))
    p_calc_test = np.array([[0 for i in range(len(calc_label_rm_duplicate))] for j in range(len(test_label_rm_duplicate))])  # 计算联合概率分布，列之和得到实际概率分布，行之和得到训练模型判断的结果概率分布
    for i in range(len(test_label)):
        p_calc_test[dic_calc[calc_label[i]]][dic_test[test_label[i]]] += 1  # 联合概率分布中，对应位置的次数加以
    p_calc_test = p_calc_test / len(test_label)
    p_calc = np.sum(p_calc_test, axis=1)  # 行之和，对于每一个标签，得到预测结果为该标签的比例
    p_test = np.sum(p_calc_test, axis=0)  # 列之和，对于每一个标签，得到测试集中该标签的比例
    h_calc = -sum([i * math.log(i, 2) for i in p_calc])  # 计算训练结果的信息熵
    h_test = -sum([i * math.log(i, 2) for i in p_test])  # 计算实际结果的信息熵
    tmp = sum([sum([p_calc_test[i][j] * (math.log(p_calc_test[i][j], 2) - math.log(p_calc[i] * p_test[j], 2)) for i in range(len(calc_label_rm_duplicate))]) for j in range(len(test_label_rm_duplicate))])
    nmi = 2 * tmp / (h_calc + h_test)
    return nmi


if __name__ == "__main__":
    Data, Label = load_file("breast.txt")  # 加载文件，得到数据样本和对应标签
    Train_Size = int(len(Data) * 0.2)  # 表示训练集的数目
    Train_Data = Data[0:Train_Size]  # 用于训练的样本
    Train_Label = Label[0:Train_Size]  # 训练的样本标签
    Test_Label = Label[Train_Size+1::]
    Cov = calc_cov(Train_Data)  # 计算协方差矩阵
    Last_NMI = 0
    for Ratio in [_/100 for _ in range(50, 100)]:
        E_Vals, E_Vecs = top_ratio(Cov, Ratio)  # 根据特征值数据分布，得到前k个最大特征值和对应的特征向量构成的特征矩阵，k根据ratio计算得到
        Data_Pac = dim_reduction(Data, E_Vecs)  # 得到降维之后的数据，包括训练集和测试集
        Train_Data_Pac = Data_Pac[0:Train_Size]  # 降维之后的训练数据
        Test_Data = Data_Pac[Train_Size+1::]  # 降维之后的测试数据
        Separated_Class = separate_by_class(Train_Data_Pac, Train_Label)
        Summary = summarize(Separated_Class)
        Calc_Label = np.array([cal_class_label(Summary, Input_Vector) for Input_Vector in Test_Data])
        NMI = get_nmi(Test_Label, Calc_Label)
        if Last_NMI != NMI:
            print("保留的特征值之和占所有特征值之和的比例: %.2f    计算所得NMI为: %.5f" % (Ratio, NMI))
        Last_NMI = NMI
