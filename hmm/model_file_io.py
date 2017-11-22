# encoding=utf-8
import hmm.hmm_variable as hmm_variable
import tool.global_variable as global_varibale
import math


# 从训练好的hmm模型文件中读取参数 init_state_prob, prob_trans_matrix, prob_emit_maps
def read_model_params_from_lines(lines):
    hmm_model_lines = lines
    init_state_prob = []
    prob_trans_matrix = []
    prob_emit_maps = []
    for index in range(len(hmm_model_lines)):
        line = hmm_model_lines[index].strip()
        # read all states
        if line.startswith(hmm_variable.model_init_state_key):
            for state_index in range(global_varibale.state_count):
                index += 1
                this_line = hmm_model_lines[index].strip()
                split_array = this_line.split(':')
                number = float(split_array[1])
                init_state_prob.append(number)
        # read prob trans matrix
        elif line.startswith(hmm_variable.model_prob_trans_key):
            for state_index in range(global_varibale.state_count):
                index += 1
                this_line = hmm_model_lines[index].strip()
                split_array = this_line.split(' ')
                prob_trans_matrix.append([])
                for str_number in split_array:
                    hmm_model_lines[index].strip()
                    number = float(str_number)
                    prob_trans_matrix[state_index].append(number)
        # read prob emit matrix, save it in a dict for better performance
        elif line.startswith(hmm_variable.model_prob_emit_key):
            for state_index in range(global_varibale.state_count):
                index += 2
                prob_emit_maps.append(dict())
                this_line = hmm_model_lines[index].strip()
                word_couple_array = this_line.split(',')
                for word_and_prob in word_couple_array:
                    if word_and_prob == '':
                        continue
                    word_and_prob_split_array = word_and_prob.strip().split(':')
                    word_as_key = word_and_prob_split_array[0].decode('utf-8')
                    prob_as_value = float(word_and_prob_split_array[1])
                    one_map = prob_emit_maps[state_index]
                    one_map[word_as_key] = prob_as_value

    return True, init_state_prob, prob_trans_matrix, prob_emit_maps


# 输出hmm训练好的参数trans_prob_matrix, emit_prob_matrix_array 输出到文件
def write_model_to_file(trans_prob_matrix, emit_prob_matrix_array, init_state_prob_array, output_path):
    output_file = open(output_path, "w+")

    # 1. write states
    output_file.write(hmm_variable.model_states_key + '\n')
    for i in range(global_varibale.state_count):
        output_file.write('#' + str(i) + ':' + global_varibale.state_array[i] + '\n')
    output_file.write('\n')

    # 2. write init_state
    # TODO how to get init states
    output_file.write(hmm_variable.model_init_state_key + '\n')
    # must start with Begin or Single, both End and Interval is zero
    init_state_array = []
    for i in init_state_prob_array:
        if i == 0:
            init_state_array.append(global_varibale.min_number)
        else:
            init_state_array.append(math.log(i))
    for i in range(len(init_state_array)):
        output_file.write('#' + global_varibale.state_array[i] + ':' + str(init_state_array[i]) + '\n')
    output_file.write('\n')

    # 3. write trans_prob_matrix
    output_file.write(hmm_variable.model_prob_trans_key + '\n')
    for i in range(global_varibale.state_count):
        for j in range(global_varibale.state_count):
            output_file.write(str(trans_prob_matrix[i][j]) + ' ')
        output_file.write('\n')
    output_file.write('\n')

    # 4. write emit_prob_matrix3
    output_file.write(hmm_variable.model_prob_emit_key + '\n')
    for i in range(global_varibale.state_count):    # hidden label: EOS
        one_map = emit_prob_matrix_array[i]
        output_file.write('#' + global_varibale.state_array[i] + '\n')
        for char, prob in one_map.items():
            output_file.write(char + ':' + str(prob) + ',')
        output_file.write('\n')

    output_file.close()
    return True, ''

