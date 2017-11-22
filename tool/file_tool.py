# encoding=utf-8
import os


def read_file_lines(file_path):
    if not os.path.isfile(file_path):
        return False, "Error: file [%s] does not exist" % file_path
    file_ptr = open(file_path, "r")
    file_content = file_ptr.read().decode("utf-8")
    lines = file_content.split('\n')
    return True, lines


# 读取训练文件的所有行,把所有char,label作为数组返回
def read_training_file(training_file_path):
    status, response = read_file_lines(training_file_path)
    if not status:
        print response
        return False, response
    lines = response
    status, input_char_list, input_state_list = read_training_lines(lines)
    return status, input_char_list, input_state_list


# 根据空行划分句子,一个句子对应一个列表
# 列表包含句子楼里面的所有char,label序列
def read_training_sentences(training_file_path):
    char_in_sentences = []
    label_of_sentences = []
    status, response = read_file_lines(training_file_path)
    if not status:
        print response
        return False, response
    lines = response

    # read test file and correct states
    sentence_char = []
    sentence_label = []
    for line in lines:
        line = line.strip()
        # 遇到一个空行,说明一个句子读取完毕
        if line == '':
            # 添加一个句子的数据到结果列表
            char_in_sentences.append(sentence_char)
            label_of_sentences.append(sentence_label)
            # 清空当前句子的缓存
            sentence_char = []
            sentence_label = []
            continue
        # 分析一行的数据
        split_array = line.split(' ')
        if len(split_array) != 2:
            print "read_training_file Line [%s] split by space is not two part" % line
            continue
        chinese_char = split_array[0].decode('utf-8')
        correct_state = split_array[1]
        sentence_char.append(chinese_char)
        sentence_label.append(correct_state)

    return True, char_in_sentences, label_of_sentences


# 读取使用空格分割的char 正确label文件作为训练数据
def read_training_lines(lines):
    input_char_list = []
    input_state_list = []

    # read test file and correct states
    for line in lines:
        line = line.strip()
        # TODO how to deal with empty line in input file
        if line == '':
            continue
        split_array = line.split(' ')
        if len(split_array) != 2:
            print "read_training_file Line [%s] split by space is not two part" % line
            continue
        chinese_char = split_array[0].decode('utf-8')
        correct_state = split_array[1]
        input_char_list.append(chinese_char)
        input_state_list.append(correct_state)

    return True, input_char_list, input_state_list
