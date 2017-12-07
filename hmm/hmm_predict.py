# encoding=utf-8
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import ai_lab_2.tool.global_variable as global_varibale
import ai_lab_2.tool.file_tool as file_tool
import numpy as np
import model_file_io as model_io


# 输入训练好的hmm模型文件路径和句子, 返回句子分割结果 state_string_result
def divide_sentence(hmm_model_file_path, sentence):
    # 0 read model file
    status, response = file_tool.read_file_lines(hmm_model_file_path)
    if not status:
        print response
        return False, response
    hmm_model_lines = response

    # 1 get model params
    status, init_state_prob, prob_trans_matrix, prob_emit_maps = model_io.read_model_params_from_lines(
        hmm_model_lines)
    if not status:
        print response
        return False, response
    init_state_prob = np.array(init_state_prob)
    prob_trans_matrix = np.array(prob_trans_matrix)

    # 2 calculate weight and path matrix for word partition
    status, weight_matrix, path_matrix = __calculate_weight_and_path(init_state_prob, prob_trans_matrix, prob_emit_maps,
                                                                     sentence)
    if not status:
        print "Error in calculate_weight_and_path"
        return False, response
    weight_matrix = np.array(weight_matrix)
    path_matrix = np.array(path_matrix)
    print weight_matrix
    print path_matrix

    # 3 get word partition by tracing back on path_matrix
    status, response = __trace_back(global_varibale.state_array, weight_matrix, path_matrix, sentence)
    if not status:
        print response
        return False, response
    returned_states_string = response

    return True, returned_states_string


def divide_sentence_in_file(hmm_model_file_path, test_file):
    # read test file
    status, response = file_tool.read_file_lines(test_file)
    if not status:
        print response
        return False, response, '', ''
    lines = response

    # read input char list and correct state list
    status, input_char_list, input_state_list = file_tool.read_training_lines(lines)
    if not status:
        print "Error in read_training_file"
        return False, '', '', ''
    origin_sentence = "".join(input_char_list)
    correct_state_string = "".join(input_state_list)

    # call divide_sentence to predict origin_sentence
    status, response = divide_sentence(hmm_model_file_path, origin_sentence)
    if not status:
        print "Error in hmm main divide_sentence ", response
    returned_states_string = response

    return True, returned_states_string, correct_state_string, origin_sentence


# Vertibi算法计算思路参考: https://yanyiwu.com/work/2014/04/07/hmm-segment-xiangjie.html
# weight: 在状态想出现一个字的概率,例如weight[0][2],在状态0也就是S下出现第2个字符'硕'的概率
# path: 对应的weight[i][j]概率最大的时候,前一个字的状态是什么
#       path[0][2] = 1, 则代表 weight[0][2]取到最大时，前一个字(也就是明)的状态是E
# return True, weight_matrix, path_matrix
def __calculate_weight_and_path(init_state_prob, prob_trans_matrix, prob_emit_maps, input_sentence):
    sentence_length = len(input_sentence)
    state_num = global_varibale.state_count

    weight_matrix = np.zeros((state_num, sentence_length))
    path_matrix = np.zeros((state_num, sentence_length))

    # init line 0 in the weight matrix
    char_index = 0
    start_char = input_sentence[char_index].decode('utf-8')
    for i in range(state_num):
        emit_map_of_this_state = prob_emit_maps[i]
        if not emit_map_of_this_state.has_key(start_char):
            weight_matrix[i][char_index] = init_state_prob[i] + global_varibale.min_number
        else:
            weight_matrix[i][char_index] = init_state_prob[i] + emit_map_of_this_state[start_char]

    # calculate all remains for each char in the sentence
    for i in range(1, sentence_length):
        # for all the states
        for state in range(state_num):
            weight_matrix[state][i] = 0 - sys.float_info.max
            path_matrix[state][i] = -1

            # 从前个状态的所有情况k分析当前状态state, 联想掷筛子 https://www.zhihu.com/question/20962240/answer/33438846
            for k in range(state_num):
                # 1 在前个状态k下,出现前一个(i-1)字符的概率
                prev_char_weight = weight_matrix[k][i - 1]
                # 2 在前个状态k下,状态转移为当前状态state的概率
                trans_to_state_prob = prob_trans_matrix[k][state]
                # 3 在当前状态state下,展现出(发射概率)当前这个字符的概率,
                # if emit_map_of_this_state do not have this key, value is 0
                this_char = input_sentence[i].decode('utf-8')
                emit_map_of_this_state = prob_emit_maps[state]
                if not emit_map_of_this_state.has_key(this_char):
                    emit_this_char_prob = global_varibale.min_number
                else:
                    emit_this_char_prob = emit_map_of_this_state[this_char]

                # 找到所有可能状态下的最大值:在state状态下,出现当前字符最可能的情况(use + because saved value is log)
                tmp = prev_char_weight + trans_to_state_prob + emit_this_char_prob
                if tmp > weight_matrix[state][i]:
                    weight_matrix[state][i] = tmp
                    path_matrix[state][i] = k

    return True, weight_matrix, path_matrix


# return True, track_back_string
def __trace_back(state_array, weight_matrix, path_matrix, sentence):
    # 对于最后一个字,判断weight矩阵的最大值对应的,状态
    last_char_label_index = -1
    last_char_index = len(sentence) - 1
    label_num = len(state_array)
    current_max_value = 0 - sys.float_info.max
    for label_index in range(label_num):
        if weight_matrix[label_index][last_char_index] > current_max_value:
            last_char_label_index = label_index

    # 返回回溯的结果
    output_state_array = [state_array[last_char_label_index]]
    print output_state_array

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
        output_state_array.append(state_array[prev_char_state])
        current_char_index -= 1
    track_back_string = "".join(reversed(output_state_array))

    # check start with B
    if track_back_string[0] != global_varibale.word_begin_char and track_back_string[0] != global_varibale.single_char:
        return False, "Error, track_back_string[0] != B/S !"
    return True, track_back_string
