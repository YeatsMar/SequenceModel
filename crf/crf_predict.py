# encoding=utf-8
import sys
import crf_train as crf_train
from ai_lab_2.tool import file_tool

reload(sys)
sys.setdefaultencoding("utf-8")


def divide_sentence(template_file, input_sentence, correct_states_str):
    pass


if __name__ == '__main__':
    base_dir = "E:\\work_file\\ai_lab_my\\"
    training_set = base_dir + "train_beijing"
    valid_label_list = [u'M', u'B', u'E']

    # 0
    valid_tag_list = [u'M', u'B', u'E']

    # 训练模型输入1: A training set of tagged sentences
    training_file = base_dir + "train.uft8"

    # 训练模型输入2: 迭代的最大次数
    parameter_T = 1000

    # 训练模型输入3: template文件-->决定了所有特征函数
    template = base_dir + "template_beijing"

    # 模型训练输出:特征函数和响应的权重
    model_output = base_dir + "crf_model_output_file"

    # model init
    crf_model = crf_train.CRF_TRAINER(training_file, template, valid_tag_list)

    # train model with sentence and valid states for T times
    crf_model.train_model(parameter_T, model_output)

    # get trained model
    status, feature_func_obj_list, func_param_vector = crf_model.get_trained_model()
    if not status:
        print "false"
        # use model to predict sentence tags
