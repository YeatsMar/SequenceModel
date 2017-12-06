# encoding=utf-8
import sys
import numpy as np
import feature_function as feature_func

min_float_value = 0 - sys.float_info.max


# linear-CRF模型维特比算法流程：
# 输入：1 模型的 K 个特征函数， 2 和对应的 k 个权重。
#      3 可能的标记个数 m, 4 观测序列 x=(x_{1},x_{2},...,x_{n}) ，
# 输出：最优标记序列 y^{*}=(y_{1}^{*},y_{2}^{*},...,y_{n}^{*})
def verbit(feature_func_list, func_param_vector, valid_label_list, input_char_list):
    sentence_length = len(input_char_list)
    label_num = len(valid_label_list)
    weight_matrix = np.zeros((label_num, sentence_length))
    path_matrix = np.zeros((label_num, sentence_length))
    # 0, 初始化,对于i=1
    for label_index in range(label_num):
        # y[0] = start
        pre_label = "label_B-1"
        # y[1] = label
        this_label = valid_label_list[label_index]
        # x = sentence, i = index
        this_feature = input_char_list[0]
        # 一个输入情况(label-1,label,x),对应的所有特征函数的和 k = 1,2,3,...,K(feature_func_num)
        feature_func_sum = feature_func.get_feature_func_sum(feature_func_list, func_param_vector, pre_label,
                                                             this_label, this_feature)
        # 得到第一个矩阵的值
        weight_matrix[label_index][0] = feature_func_sum
        # 第二个path矩阵直接初始化为起点
        path_matrix[label_index][0] = -1

    # 1, 对于i = 1,2,...,n-1进行递推 n = sentence_length
    for i in range(1, sentence_length):
        # 对于所有的l = 1,2,3,...,m. m = label_num
        for label_index in range(label_num):
            # 对于所有的label进行计算,找出其中的最大值,作为weight[label_index][i]的数值, 1 <= j <= m
            weight_matrix[label_index][i] = min_float_value
            path_matrix[label_index][i] = -1
            for j in range(label_num):
                # y[i-1] = j
                prev_value = weight_matrix[j][i - 1]
                # 计算当label是l, y[i-1]=j, y[i]=l, x, i的所有特征函数的和
                pre_label = valid_label_list[j]
                this_label = valid_label_list[label_index]
                this_feature = input_char_list[i]
                feature_func_sum = feature_func.get_feature_func_sum(feature_func_list, func_param_vector,
                                                                     pre_label, this_label, this_feature)
                # 找到所有可能状态下的最大值
                tmp = prev_value + feature_func_sum
                if tmp > weight_matrix[label_index][i]:
                    weight_matrix[label_index][i] = tmp
                    path_matrix[label_index][i] = j

    status, response = __trace_back(valid_label_list, weight_matrix, path_matrix, input_char_list)
    if not status:
        print "Error in __trace_back"
    output_state_array = response
    return True, output_state_array[::-1]


# return True, output_state_array
def __trace_back(label_array, weight_matrix, path_matrix, input_array):
    # 对于最后一个字,判断weight矩阵的最大值对应的,状态
    last_char_label_index = 0
    last_char_index = len(input_array) - 1
    label_num = len(label_array)
    current_max_value = min_float_value
    for label_index in range(label_num):
        tmp = weight_matrix[label_index][last_char_index]
        if tmp > current_max_value:
            last_char_label_index = label_index
            current_max_value = tmp

    # 返回回溯的结果
    output_state_array = [label_array[last_char_label_index]]

    # 遍历所有字符, 从path数组回溯所有前面字符的状态
    current_char_index = last_char_index
    prev_char_state = last_char_label_index
    while current_char_index >= 1:
        # 当前字符的状态, 是上一次循环中,后面一个字符决定的
        current_char_state = prev_char_state
        # 获取当前字符,当前状态对应的path数组信息
        current_path_matrix_value = int(path_matrix[current_char_state][current_char_index])
        # path: 对应的weight[i][j]概率最大的时候,前一个字的状态是什么
        prev_char_state = int(current_path_matrix_value)
        output_state_array.append(label_array[prev_char_state])
        current_char_index -= 1

    return True, output_state_array
