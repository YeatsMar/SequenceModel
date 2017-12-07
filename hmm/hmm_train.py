# encoding=utf-8
import sys
import math

reload(sys)
sys.setdefaultencoding("utf-8")

import ai_lab_2.tool.file_tool as file_tool
import ai_lab_2.tool.global_variable as global_varibale
import numpy as np
import model_file_io as model_io


def train_hmm_model(training_data_path, output_path):
    # read model file content
    status, response = file_tool.read_file_lines(training_data_path)
    if not status:
        print response
        return False, response
    lines = response

    # get input char list and corresponding state list
    status,  input_char_list, correct_state_list = file_tool.read_training_lines(lines)
    if not status:
        return False, "Error in __read_training_set"

    # calculate basic word frequency to get params
    status, trans_prob_matrix, emit_prob_matrix_array = __train_model(input_char_list, correct_state_list)
    if not status:
        print "Error in get_init_params"
        return False, response

    # write model to file in specific format
    status, response = model_io.write_model_to_file(trans_prob_matrix, emit_prob_matrix_array, output_path)
    if not status:
        print response
        return False, response

    return True, ''


def __train_model(input_char_list, correct_state_list):
    trans_count_matrix = np.zeros((global_varibale.state_count, global_varibale.state_count))
    emit_count_dict_array = []
    state_count_array = []
    for i in range(global_varibale.state_count):
        # 每个状态对应的, prob_trans_dict
        emit_count_dict_array.append(dict())
        # 每个状态出现的次数
        state_count_array.append(0)

    length = len(input_char_list) - 1
    total_char_count = len(input_char_list)
    for i in range(length):
        this_char = input_char_list[i]
        this_state = correct_state_list[i]
        next_state = correct_state_list[i + 1]
        this_state_index = global_varibale.state_array.index(this_state)
        next_state_index = global_varibale.state_array.index(next_state)

        # 1 state change: this state --> next state
        trans_count_matrix[this_state_index][next_state_index] += 1

        # 2 count this char under this state(emit)
        state_prob_emit_dict = emit_count_dict_array[this_state_index]
        state_count_array[this_state_index] += 1
        if state_prob_emit_dict.has_key(this_char):
            state_prob_emit_dict[this_char] += 1
        else:
            state_prob_emit_dict[this_char] = 1

    # TODO how to handle last char
    last_char = input_char_list[length]
    last_char_state = correct_state_list[length]
    last_char_index = global_varibale.state_array.index(last_char_state)
    state_prob_emit_dict = emit_count_dict_array[last_char_index]
    state_count_array[last_char_index] += 1
    if state_prob_emit_dict.has_key(last_char):
        state_prob_emit_dict[last_char] += 1
    else:
        state_prob_emit_dict[last_char] = 1

    # check sum
    status, response = __check_sum(state_count_array, emit_count_dict_array, trans_count_matrix, total_char_count)
    if not status:
        print response
        return False, '', ''

    # calculate frequency
    trans_prob_matrix = np.zeros((4, 4))
    emit_prob_matrix_array = []
    for i in range(global_varibale.state_count):
        this_state_count = state_count_array[i]
        one_map = emit_count_dict_array[i]

        # 计算状态转移矩阵
        for j in range(global_varibale.state_count):
            count = trans_count_matrix[i][j]
            if count == 0:
                trans_prob_matrix[i][j] = global_varibale.min_number
                continue
            p = count / this_state_count
            trans_prob_matrix[i][j] = math.log(p)

        # 计算每一个状态对应的Word map
        one_emit_prob_matrix = dict()
        for char, char_count in one_map.items():
            one_emit_prob_matrix[char] = math.log((char_count + 0.0) / this_state_count)
        emit_prob_matrix_array.append(one_emit_prob_matrix)

    return True, trans_prob_matrix, emit_prob_matrix_array


def __check_sum(state_count_array, emit_count_dict_array, trans_count_matrix, total_char_count):
    # state count sum == char number
    if sum(state_count_array) != total_char_count:
        return False, "sum(state_count_array) != total_char_count"

    emit_dict_sum = 0
    for i in range(global_varibale.state_count):
        one_map = emit_count_dict_array[i]
        emit_dict_sum += sum(one_map.values())
    # emit dict count sum == char number
    if emit_dict_sum != total_char_count:
        return False, "sum(state_count_array) != total_char_count"

    # state trans count == char number - 1 (last state cannot trans to another state)
    if int(np.sum(trans_count_matrix)) != total_char_count - 1:
        return False, "sum(prob_trans_count_matrix) != total_char_count - 1"
    return True, ''


if __name__ == '__main__':
    training_data_path = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\train10"
    output_path = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\lab_train_data.dat"
    train_hmm_model(training_data_path, output_path)
