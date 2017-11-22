# encoding=utf-8

# 根据分割结果BE/BE/BME/BE/BME/BE/S转换输入的字符串为小明/硕士/毕业于/中国/科学院/计算/所
from ai_lab_2.tool import global_variable


def calculate_accuracy(state_string_result, correct_state_string):
    if len(state_string_result) != len(correct_state_string):
        return False, "len(state_string_result) != len(correct_state_string)"

    total_count = len(state_string_result)
    same_count = 0
    for i in range(total_count):
        if state_string_result[i] == correct_state_string[i]:
            same_count += 1

    return True, (same_count + 0.0) / total_count


def process_state_array_result(returned_states_string, origin_sentence):
    readable_result = []
    for i in range(len(returned_states_string)):
        this_char = returned_states_string[i]
        readable_result.append(origin_sentence[i])
        if this_char == global_variable.word_end_char or this_char == global_variable.single_char:
            readable_result.append('/')
    return True, "".join(readable_result)


def print_predict_result(returned_states_string, correct_states_string, input_sentence):
    print "returned_state_string = ", returned_states_string
    print "correct_state_string  = ", correct_states_string

    # process result to readable format
    status, response = process_state_array_result(returned_states_string, input_sentence)
    if not status:
        print response
        return False, response
    print "origin_sentence  = ", input_sentence
    print " readable_result = ", response

    # calculate and print predict accuracy
    status, response = calculate_accuracy(returned_states_string, correct_states_string)
    if not status:
        print response
        return False, response
    print "calculate_accuracy = ", response

    return True, ''
