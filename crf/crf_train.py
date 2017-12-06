# encoding=utf-8
import sys
import numpy as np

import template_reader as template_reader
import feature_function as feature_func
import viterbi_function as viterbi_func
import input_helper as my_input_helper
import tool.model_tool as model_tool
import tool.global_variable as global_variable


reload(sys)
sys.setdefaultencoding("utf-8")


class CRF_TRAINER:
    def __init__(self, training_file, template_file, valid_tag_list):
        # 模型的输入
        self.training_file = training_file
        self.template_file = template_file
        self.valid_tag_list = valid_tag_list
        self.all_char_list = []
        self.all_state_list = []

        # 两个辅助处理类
        self.template_reader = template_reader.TemplateReader()
        self.input_helper = my_input_helper.InputHelper(training_file)

        # 处理输入模板的处理结果,所有的特征函数和对应的参数信息
        self.feature_func_obj_list = []
        self.func_param_vector = ''

    # ------------------------------ public function ------------------------------
    def train_model(self, parameter_t):
        # 0 从训练集文件中获取所有的信息
        status, sentences, label_of_sentences, all_char_list, all_state_list = \
            self.input_helper.read_training_file()
        if not status:
            print "Error in read_training_file"
            return False, '', ''
        self.all_char_list = all_char_list
        self.all_state_list = all_state_list

        # 1 模型与特征函数的初始化,获取所有的特征函数
        status, response = self.__model_init(all_char_list, all_state_list)
        if not status:
            print response
            return False, response, ''
        self.feature_func_obj_list = response

        # 2 模型训练,获取所有特征函数的权重
        status, response = self.__start_training(parameter_t, sentences, label_of_sentences)
        if not status:
            print response
            return False, response, ''
        self.func_param_vector = response

        # 3 返回训练结果,特征函数与权重
        return True, self.feature_func_obj_list, self.func_param_vector

    # ------------------------------ local function ------------------------------
    def __model_init(self, input_char_list, correct_label_list):
        # 根据所有char和label,初始化训练输入矩阵
        status, response = self.input_helper.init_training_matrix(input_char_list, correct_label_list)
        if not status:
            print "Error in init_training_matrix"
            return False, ''

        # 读取模板文件,获取所有输入macro
        status, response = self.template_reader.read(self.template_file)
        if not status:
            print "Error in reader.read"
            return False, ''
        all_macros = self.template_reader.get_all_macros()

        # 根据模板里面的宏定义遍历训练集training_matrix,获取每一个模板扩展出来的字符串/每一个模板对应的输入macro信息
        input_len = len(input_char_list)
        status, template_feature_dict, template_macro_dict = self.input_helper.get_input_macro(all_macros,
                                                                                               input_len)
        if not status:
            print "Error in get_input_macro"
            return False, ''

        # 遍历模板扩展出来的字符串/每一个模板对应的输入macro信息,获取所有特征函数
        status, response = self.__get_feature_func_list(template_feature_dict, template_macro_dict)
        if not status:
            print "Error in get_feature_func_obj"
            return False, response
        feature_func_obj_list = response
        return True, feature_func_obj_list

    def __start_training(self, training_t, sentences, label_of_sentences):
        # input 1 feature functions is ready after model init
        print "input 1, total feature func number = ", len(self.feature_func_obj_list)
        # input 2, set init feature function param vector to 0
        input_param_vector = np.zeros(len(self.feature_func_obj_list))

        # 训练总共迭代T次 TODO 每一次迭代打印准确率
        for t in range(training_t):
            sentence_num = len(sentences)
            if sentence_num <= 0:
                print "input sentence is empty, error"
                return False, ''
            # 本次迭代开始时候的参数:上一次迭代的结果
            param_of_one_iteration = np.copy(input_param_vector)

            # 一次迭代中,对于所有的句字,计算当下模型输出,并调整参数
            for i in range(sentence_num):
                # 根据句子下标获取对应的句子和正确的label
                sentence = sentences[i]
                correct_tag_list = label_of_sentences[i]
                # 根据句子下标获取多行多列的输入矩阵,这里只有两列
                status, response = self.input_helper.get_input_matrix_of_sentence(i)
                if not status:
                    print "Error in get_input_matrix"
                    return False, response
                input_matrix = response

                # 对这个句子进行一次迭代
                status, response = self.__iterate_sentence(self.feature_func_obj_list, param_of_one_iteration,
                                                           self.valid_tag_list, sentence, correct_tag_list,
                                                           input_matrix)
                if not status:
                    print "Error in iteration" + str(t) + response
                    return False, response
                # 调整参数, 对下一个句子进行迭代
                param_of_one_iteration = response

            # 一次迭代结束,打印结果
            self.__print_accuracy(t, sentences, param_of_one_iteration)
            print "equal = ", np.equal(param_of_one_iteration, input_param_vector)

            # 如果参数不再变化,模型收敛,结束迭代
            if np.array_equal(param_of_one_iteration, input_param_vector):
                print "Param remains the same"
                final_func_param_vector = input_param_vector
                return True, final_func_param_vector
            # 参数有变化,调整参数,进入下一次迭代
            input_param_vector = param_of_one_iteration

        # 迭代结束,训练完成
        final_func_param_vector = input_param_vector
        return True, final_func_param_vector

    def __iterate_sentence(self, feature_func_list, func_param_vector, valid_label_list,
                           input_sentence, correct_tag_list, input_matrix):
        # Use the Viterbi algorithm to find the output of the model on the i'th sentence with current parameter
        status, response = viterbi_func.verbit(feature_func_list, func_param_vector, valid_label_list, input_sentence)
        if not status:
            return False, response
        output_state_array = response

        # 拓展输入的句子前后
        output_state_string = "".join(output_state_array)
        correct_label_string = "".join(correct_tag_list)
        status, extend_char_list, ex_output_list = self.input_helper.get_extend_input_array(input_sentence,
                                                                                            output_state_string)
        if not status:
            return False, ''
        status, extend_char_list, ex_correct_list = self.input_helper.get_extend_input_array(input_sentence,
                                                                                             correct_label_string)
        if not status:
            return False, ''
        func_param_copy = np.copy(func_param_vector)

        #  If z[1:ni](output_state_array) != ti[1:ni](correct_state) then update the parameters
        if output_state_string != correct_label_string:
            # 调整每一个参数
            for s in range(len(feature_func_list)):
                this_feature_func = feature_func_list[s]
                # 遍历所有输入的单词,对每一个单词计算这个特征函数的和
                sum_of_correct = 0.0
                sum_of_current_best = 0.0
                for i in range(len(input_sentence)):
                    extend_i_index = self.input_helper.get_extend_input_index() + i
                    # 需要根据函数的macro计算输入的feature = [U03:jing]  or [U05:bei/jing]
                    input_macro = this_feature_func.get_macro_dict()
                    # 一个句子对应一个training_matrix,都在头和尾,input_matrix是被拓展的
                    this_feature = self.input_helper.get_input_feature_by_macro(input_macro, input_matrix,
                                                                                extend_i_index)

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
            print "output_state_string == correct_label_string:[%s]" % output_state_string

        return True, func_param_copy

    # template_str_feature_dict:一个模板对应的训练文件中所有的特征字符串list
    # template_macro_dict:一个模板对应的输入macro信息,例如%x[-1,0]
    def __get_feature_func_list(self, template_feature_dict, template_macro_dict):
        template_id_list = self.template_reader.get_template_id_list()
        valid_tags = self.valid_tag_list
        feature_func_obj_list = []
        # 1 前L * L个特征函数是状态转移矩阵,也就是f(s',s,input=null),只有前一个和这一个的状态
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

        # 2 unigram生成的特征函数的数目 = (L * N)
        # 3 bigram生成的特征函数的数目 = (L * L * N)
        # N是通过模板扩展出来的所有单个字符串(特征）的个数
        # L是合法label的数目,这里就是BEIS 4个
        for template_id in template_id_list:
            # 这个template获取的输入文件中所有的字符串特征
            feature_list = template_feature_dict[template_id]
            # 这个template对应的输入macro
            input_macro = template_macro_dict[template_id]
            # 对于每一个特征字符串,生成对应的特征函数,例如一个feature可以是 U01:北
            for feature in feature_list:
                # 跳过那些被拓展的feature,例如 U01:_B-1 是被 U01:%x[-1,0] 拓展出来的
                if '_B' in feature:
                    continue
                # 这个特征的unigram : Label + input（当前feature字符串）
                if template_id.startswith('U'):
                    for one_tag in valid_tags:
                        params = {
                            "prev_label": '',
                            "label": one_tag,
                            "feature": feature,
                            'macro_dict': input_macro
                        }
                        feature_func_obj = feature_func.FeatureFunction(params)
                        feature_func_obj_list.append(feature_func_obj)

                # 这个特征的bigram Label(i-1) + label + input
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

    def __print_accuracy(self, t, sentences, input_param_vector):
        print "Iteration num : [%d]" % t
        print "Print accuracy"

        # use model to predict sentence tags
        all_returned_state = ""
        for sentence in sentences:
            status, result = viterbi_func.verbit(self.feature_func_obj_list, input_param_vector,
                                                 global_variable.state_array, sentence)
            if not status:
                print "Error in verbit"
                return False, ''
            all_returned_state += "".join(result)

        all_correct_state_str = "".join(self.all_state_list)
        all_char_list_str = "".join(self.all_char_list)
        st, resp = model_tool.print_predict_result(all_returned_state, all_correct_state_str, all_char_list_str)
        if not st:
            print resp
            return False, ''
        return True, ''