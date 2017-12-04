# encoding=utf-8
import sys
import math
import operator
import numpy as np

import template_reader as template_reader
import feature_function as feature_func
import ai_lab_2.tool.global_variable as global_variable
from ai_lab_2.tool import file_tool

reload(sys)
sys.setdefaultencoding("utf-8")


class CRF_TRAINER:
    def __init__(self, training_file, template_file, valid_tag_list):
        # 模型的输入
        self.training_file = training_file
        self.template_file = template_file
        self.valid_tag_list = valid_tag_list

        # 其他参数
        # 在训练集的第一行之前增加2行不存在的文字,用于处理第一行的特殊情况: %x[-2, 0]
        self.extend_input_index = 2
        self.training_set_size = 0
        self.macro_split_char = '/'
        self.id_macro_split_char = ':'
        self.reader = template_reader.TemplateReader()

        # 处理输入模板的处理结果
        self.all_macros = []
        self.feature_func_obj_list = []
        self.func_param_vector = ''

        # 输入文件中所有的句子和label
        self.sentences = []
        self.label_of_sentences = []
        # 所有输入字符的大list
        self.input_char_list = []
        self.correct_label_list = []
        # 在输入文件所有字符前后增加2个字符的训练输入矩阵
        self.training_matrix = []
        # 每一个句子对应的训练输入矩阵
        self.training_matrix_for_sentence = {}

        # 一个template对应的所有string feature list
        self.template_str_feature_dict = {}
        # 一个template对应的一个输入macro
        self.template_macro_dict = {}

        # 标记模型是否已经训练
        self.is_trained = False

    def model_init(self):
        # 读取模板文件,获取所有输入macro
        self.reader.read(self.template_file)
        self.all_macros = self.reader.get_all_macros()

        # 读取训练文件内容1:读取所有句子和对应的标记
        status, sentences, label_of_sentences = file_tool.read_training_sentences(self.training_file)
        if not status:
            print "Error when read_training_sentences"
            return False, ''
        print "sentence count = [%d]" % len(sentences)
        self.sentences = sentences
        self.label_of_sentences = label_of_sentences

        # 读取训练文件内容2:读取所有char和label到一个大list,以便获取所有特征函数
        status, all_char_list, all_state_list = file_tool.read_training_file(training_set)
        if not status:
            print "Error in read_training_file"
            return False, ''
        print "total char count = [%d]" % len(all_char_list)
        self.input_char_list = all_char_list
        self.correct_label_list = all_state_list

        # 根据所有char和label,初始化训练输入矩阵
        status, response = self.init_training_matrix(all_char_list, all_state_list)
        if not status:
            print "Error in init_training_matrix"
            return False, ''
        one_training_matrix = response
        print "one_training_matrix = ", one_training_matrix
        self.training_matrix = one_training_matrix

        # 根据模板里面的宏定义遍历训练集,获取所有通过模板扩展出来的字符串
        status, template_str_feature_dict, template_macro_dict = self.get_input_macro(self.all_macros)
        if not status:
            print "Error in get_input_macro"
            return False, ''
        self.template_str_feature_dict = template_str_feature_dict
        self.template_macro_dict = template_macro_dict

        # 遍历一次训练集,获取所有特征函数
        status, response = self.get_feature_func_obj()
        if not status:
            print "Error in get_feature_func_obj"
            return False, response
        self.feature_func_obj_list = response

        return True, ''

    def train_model(self, parameter_t, model_output_file):
        # 模型与特征函数的初始化
        self.model_init()

        # 模型训练
        self.start_training(parameter_t)

        # 输出模型结果,以便后面使用模型预测
        model_output_ptr = open(model_output_file, "w+")
        for i in range(len(self.feature_func_obj_list)):
            func_obj = self.feature_func_obj_list[i]
            model_output_ptr.write(str(i) + "\t" + func_obj.to_string() + '\n')
        model_output_ptr.close()
        self.is_trained = True
        return True, ''

    def start_training(self, training_t):
        print "self.parameter_T = ", training_t
        # input 1 feature functions is ready after model init
        print "input 1, total feature func number = ", len(self.feature_func_obj_list)
        # input 2,init feature function param vector
        input_param_vector = np.zeros(len(self.feature_func_obj_list))

        # 训练总共迭代T次
        for t in range(training_t):
            print "Iteration num : [%d]" % t
            # 一次迭代中,对于所有的句字,计算当下模型输出,并调整参数
            for i in range(len(self.sentences)):
                # 根据句子下表获取对应的句子和正确的label
                sentence = self.sentences[i]
                correct_tag_list = self.label_of_sentences[i]
                print "process sentence = [%s]", sentence
                # 根据句子下表获取多行多列的输入矩阵,这里只有两列
                status, response = self.get_input_matrix_of_sentence(i)
                if not status:
                    print "Error in get_input_matrix"
                    return False, response
                input_matrix = response
                # 对这个句子进行迭代
                status, response = self.iterate_sentence(self.feature_func_obj_list, input_param_vector,
                                                         self.valid_tag_list, sentence, correct_tag_list, input_matrix)
                if not status:
                    print "Error in iteration" + str(t) + response
                    return False, response
                # 调整参数
                new_param_vector = response
                # 下面对下一个句子进行迭代

            # 迭代完所有的句子, 如果参数不再变化,迭代结束
            if np.array_equal(input_param_vector, new_param_vector):
                print "func_param_vector stays the same, end iteration now"
                break
            input_param_vector = new_param_vector

        # 迭代结束,训练完成
        self.func_param_vector = input_param_vector
        print "training end, func_param_vector = ", self.func_param_vector
        return True, ''

    def get_input_matrix_of_sentence(self, sentence_index):
        # 已经计算过这个句子对应的输入矩阵
        if self.training_matrix_for_sentence.has_key(str(sentence_index)):
            return self.training_matrix_for_sentence[sentence_index]
        # 计算此句子对应的input_matrix
        sentence = self.sentences[sentence_index]
        label_array = self.label_of_sentences[sentence_index]

        status, extend_char_list, ex_output_list = self.get_extend_input_array(sentence, label_array)
        if not status:
            return False, ''

        new_input_matrix = np.vstack((extend_char_list, ex_output_list))
        new_input_matrix = np.transpose(new_input_matrix)
        print "extend_char_list = ", extend_char_list
        print "ex_output_list = ", ex_output_list
        print "new_input_matrix = ", new_input_matrix

        self.training_matrix_for_sentence[sentence_index] = new_input_matrix
        return new_input_matrix

    def get_trained_model(self):
        if not self.is_trained:
            return False, "please call train() func to train the model first", ''
        else:
            return True, self.feature_func_obj_list, self.func_param_vector

    def iterate_sentence(self, feature_func_list, func_param_vector, valid_label_list,
                         input_sentence, correct_tag_list, input_matrix):
        # Use the Viterbi algorithm to find the output of the model
        # on the i'th training sentence with the current parameter settings
        status, response = self.verbit(feature_func_list, func_param_vector, valid_label_list, input_sentence)
        if not status:
            return False, response
        output_state_array = response

        # 拓展输入的句子前后
        output_state_string = "".join(output_state_array)
        correct_label_string = "".join(correct_tag_list)
        status, extend_char_list, ex_output_list = self.get_extend_input_array(input_sentence, output_state_string)
        if not status:
            return False, ''
        status, extend_char_list, ex_correct_list = self.get_extend_input_array(input_sentence, correct_label_string)
        if not status:
            return False, ''
        func_param_copy = np.copy(func_param_vector)

        #  If z[1:ni](output_state_array) != ti[1:ni](correct_state) then update the parameters
        if output_state_string != correct_label_string:
            print "ex_output_list = ", ex_output_list
            # 调整每一个参数
            for s in range(len(feature_func_list)):
                this_feature_func = feature_func_list[s]
                # 遍历所有输入的单词,对每一个单词计算这个特征函数的和
                sum_of_correct = 0.0
                sum_of_current_best = 0.0
                for i in range(len(input_sentence)):
                    extend_i_index = self.extend_input_index + i
                    # 需要根据函数的macro计算输入的feature = [U03:jing]  or [U05:bei/jing]
                    input_macro = this_feature_func.get_macro_dict()
                    # 一个句子对应一个training_matrix,都在头和尾
                    this_feature = self.get_input_feature_by_macro(input_macro, input_matrix, extend_i_index)

                    # 正确的label序列是correct_label_string
                    pre_label = ex_correct_list[extend_i_index - 1]
                    this_label = ex_correct_list[extend_i_index]
                    sum_of_correct += this_feature_func.calculate(pre_label, this_label, this_feature)

                    # 本次迭代中,获取的最佳序列
                    pre_label = ex_output_list[extend_i_index - 1]
                    this_label = ex_output_list[extend_i_index]
                    sum_of_current_best += this_feature_func.calculate(pre_label, this_label, this_feature)

                # 调整一个参数: 在旧参数的基础上 + 正确序列函数求和 - 当前最佳序列函数求和
                func_param_copy[s] = func_param_copy[s] + sum_of_correct - sum_of_current_best
        # 当前模型已经可以产出正确结果,无需更新模型参数
        else:
            print "output_state_string == correct_label_string, model is good now"

        return True, func_param_copy

    # linear-CRF模型维特比算法流程：
    # 输入：1 模型的 K 个特征函数， 2 和对应的 k 个权重。
    #      3 可能的标记个数 m, 4 观测序列 x=(x_{1},x_{2},...,x_{n}) ，
    # 输出：最优标记序列 y^{*}=(y_{1}^{*},y_{2}^{*},...,y_{n}^{*})
    def verbit(self, feature_func_list, func_param_vector, valid_label_list, input_char_list):
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
            feature_func_sum = self.get_feature_func_sum(feature_func_list, func_param_vector, pre_label, this_label,
                                                         this_feature)
            # 得到第一个矩阵的值
            weight_matrix[label_index][0] = feature_func_sum
            # 第二个path矩阵直接初始化为起点
            path_matrix[label_index][0] = -1

        # 1, 对于i = 1,2,...,n-1进行递推 n = sentence_length
        for i in range(1, sentence_length):
            # 对于所有的l = 1,2,3,...,m. m = label_num
            for label_index in range(label_num):
                # 对于所有的label进行计算,找出其中的最大值,作为weight[label_index][i]的数值, 1 <= j <= m
                weight_matrix[label_index][i] = 0 - sys.float_info.max
                path_matrix[label_index][i] = -1
                for j in range(label_num):
                    # y[i-1] = j
                    prev_value = weight_matrix[j][i - 1]
                    # 计算当label是l, y[i-1]=j, y[i]=l, x, i的所有特征函数的和
                    pre_label = valid_label_list[j]
                    this_label = valid_label_list[label_index]
                    this_feature = input_char_list[i]
                    feature_func_sum = self.get_feature_func_sum(feature_func_list, func_param_vector, pre_label,
                                                                 this_label, this_feature)

                    # 找到所有可能状态下的最大值
                    tmp = prev_value + feature_func_sum
                    if tmp > weight_matrix[label_index][i]:
                        weight_matrix[label_index][i] = tmp
                        path_matrix[label_index][i] = j

        status, response = self.__trace_back(valid_label_list, weight_matrix, path_matrix, input_char_list)
        if not status:
            print "Error in __trace_back"
        output_state_array = response
        return True, reversed(output_state_array)

    # return True, output_state_array
    def __trace_back(self, label_array, weight_matrix, path_matrix, input_array):
        # 对于最后一个字,判断weight矩阵的最大值对应的,状态
        last_char_label_index = 0
        last_char_index = len(input_array) - 1
        label_num = len(label_array)
        current_max_value = 0 - sys.float_info.max
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

    def get_feature_func_sum(self, feature_func_list, func_param_vector, pre_label, this_label, this_feature):
        feature_func_num = len(feature_func_list)
        feature_func_sum = 0.0
        for k in range(feature_func_num):
            func = feature_func_list[k]
            func_param = func_param_vector[k]
            func_result = func_param * func.calculate(pre_label, this_label, this_feature)
            feature_func_sum += func_result
        return feature_func_sum

    def get_extend_input_array(self, input_char_list, input_state_list):
        # 在首尾添加输入的拓展训练集
        extend_input_char_list = ["_B-2", "_B-1"]
        for input_char in input_char_list:
            extend_input_char_list.append(input_char)
        extend_input_char_list.append("_B+1")
        extend_input_char_list.append("_B+2")

        extend_correct_label_list = ["label_B-2", "label_B-1"]
        for input_state in input_state_list:
            extend_correct_label_list.append(input_state)
        extend_correct_label_list.append("label_B+1")
        extend_correct_label_list.append("label_B+2")

        if len(extend_correct_label_list) != len(self.input_char_list) + 2 * self.extend_input_index:
            print "get_extend_input_array false"
            return False, '', ''
        return True, extend_input_char_list, extend_correct_label_list

    # 输入的第一列:原始char的转置
    # 输入的第二列:label的转置,CRF模型可以支持多列,这里只有两列
    # 用来被macro根据下表处理的输入矩阵 training_matrix
    def init_training_matrix(self, input_char_list, input_state_list):
        status, extend_char_list, extend_label_list = self.get_extend_input_array(input_char_list, input_state_list)
        if not status:
            return False, ''
        training_matrix = []
        # 把首尾增加了输入的list加入training_matrix
        for i in range(len(extend_char_list)):
            # x对应哪一行,y对应列,这里只有n行,2列. 第一列是文字,第二列是label
            one_training_example = [extend_char_list[i], extend_label_list[i]]
            training_matrix.append(one_training_example)
        return True, training_matrix

    def get_input_macro(self, all_macros):
        template_str_feature_dict = {}
        template_macro_dict = {}
        # 遍历一次训练集,获取所有特征函数
        for i in range(self.training_set_size):
            this_char_index = i + self.extend_input_index
            # 遍历所有模板,一个输入对应的所有特征函数的特征
            for one_macro in all_macros:
                template_id = one_macro.get("id")
                # 获取当前macro定位到的输入feature
                feature_for_this_line = self.get_input_feature_by_macro(one_macro, self.training_matrix,
                                                                        this_char_index)
                # 一个macro对应的所有字符串feature,为了后面计算每一行template对应的所有特征函数保留
                if not template_str_feature_dict.has_key(template_id):
                    template_str_feature_dict[template_id] = [feature_for_this_line]
                else:
                    template_str_feature_dict[template_id].append(feature_for_this_line)
                # 一个template对应的一个输入macro
                if not template_macro_dict.has_key(template_id):
                    template_macro_dict[template_id] = one_macro

        return True, template_str_feature_dict, template_macro_dict

    # 根据macro和当前的位置,获取training_matrix里面需要的feature
    # 例子input_macro = %x[-1,0], current_index(i) = 3, 则获取的feature就是
    # training_matrix里面前一个index(i=2)对应的输入char
    # current_index是加了前后2两个不存在的label之后的下标,最小的是2,所以当x是-2的时候,获取的x就是matrix里面的第0个,就是B:x-2
    def get_input_feature_by_macro(self, input_macro, training_matrix, current_index):
        # 如果input_macro是空,对应的是状态转移特征函数,输入的当前feature是空
        if not input_macro:
            return ''

        # 根据macro在training_matrix里面获取需要的feature
        template_id = input_macro["id"]
        macro_xy_list = input_macro["xy_list"]
        feature_of_macro = []
        for xy_item in macro_xy_list:
            x = xy_item[0] + current_index
            y = xy_item[1]
            # unigram的feature是(当前label+当前输入),例如(B,U02:北)
            # 对应的一个特征函数就是 U02 func1 = if (output = B and feature="U02:北") return 1 else return 0 
            feature_of_macro.append(training_matrix[x][y])
            feature_of_macro.append(self.macro_split_char)
        # 去除末尾的/
        feature_of_macro = feature_of_macro[:-1]

        # U01和U02这些标志位，与特征token组合到一起主要是区分“U01:問”和“U02:問”这类特征
        # 虽然抽取的"字"特征是一样的，但是在CRF中这是有区别的特征。
        feature_for_this_line = template_id + self.id_macro_split_char + "".join(feature_of_macro)
        return feature_for_this_line

    def get_feature_func_obj(self):
        template_id_list = self.reader.get_template_id_list()
        valid_tags = self.valid_tag_list
        feature_func_obj_list = []
        # 前L * L个特征函数是状态转移矩阵,也就是f(s',s,input=null),只有前一个和这一个的状态
        for one_tag in valid_tags:
            for another_tag in valid_tags:
                params = {
                    "prev_label": one_tag,
                    "label": another_tag,
                    "feature": '',  # no feature, no input macro
                    "macro_dict": ''
                }
                feature_func_obj = feature_func.FeatureFunction(params)
                feature_func_obj_list.append(feature_func_obj)

        # unigram生成的特征函数的数目 = (L * N)
        # bigram生成的特征函数的数目 = (L * L * N)
        for template_id in template_id_list:
            # N是通过模板扩展出来的所有单个字符串(特征）的个数
            feature_info = self.template_str_feature_dict[template_id]
            # 这个template对应的输入macro
            input_macro = self.template_macro_dict[template_id]
            # 对于每一个特征字符串,生成对应的特征函数
            for feature in feature_info:
                # unigram : Label + input
                if template_id.startswith('U'):
                    # L是输出的类型的个数，这里是tag
                    for one_tag in valid_tags:
                        params = {
                            "prev_label": '',
                            "label": one_tag,
                            "feature": feature,
                            'macro_dict': input_macro
                        }
                        feature_func_obj = feature_func.FeatureFunction(params)
                        feature_func_obj_list.append(feature_func_obj)
                        # 结果集的遍历L结束
                # bigram Label(i-1) + label + input
                elif template_id.startswith('B'):
                    # 两个结果集的遍历L*L结束
                    for one_tag in valid_tags:
                        for another_tag in valid_tags:
                            params = {
                                "prev_label": one_tag,
                                "label": another_tag,
                                "feature": feature,
                                'macro_dict': input_macro
                            }
                            feature_func_obj = feature_func.FeatureFunction(params)
                            feature_func_obj_list.append(feature_func_obj)
        # 一个模板结束
        return True, feature_func_obj_list


if __name__ == '__main__':
    base_dir = "E:\\work_file\\ai_lab_my\\"
    training_set = base_dir + "train_beijing"
    template = base_dir + "template_beijing"
    model_output = base_dir + "crf_model_output_file"

    crf_train = CRF_TRAINER(training_set, template, model_output)
    # test the case found in zhihu
    crf_train.train_model()
