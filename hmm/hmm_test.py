# encoding=utf-8
import hmm_predict as hmm_model
import hmm_train as hmm_train
import ai_lab_2.tool.model_tool as model_tool


# 网上找的的一个训练好的模型数据,直接返回一个句子的分词结果
def test_prediction(model_file_path, input_sentence, correct_states_str):
    # partition a sentence
    st, resp = hmm_model.divide_sentence(model_file_path, input_sentence)
    if not st:
        print "Error in hmm main ", resp
    returned_states_string = resp

    st, resp = model_tool.print_predict_result(returned_states_string, correct_states_str, input_sentence)
    if not st:
        print resp
    print "TEST END"


# 根据train.uft8简单统计词频的输入数据
def test_train_and_predict(training_data_path, output_model_file_path, test_data_file):
    # train data model --> save model to output_model_file_path
    print "####### train data model start ######## "
    hmm_train.train_hmm_model(training_data_path, output_model_file_path)
    print "####### train data model done ######## "

    # use output_model to predict test data: test_data_path
    print "####### predict start ######## "
    st, returned_state_string, correct_state_string, origin_sentence = hmm_model.divide_sentence_in_file(
        output_model_file_path, test_data_file)
    if not st:
        print "Error in hmm main ", returned_state_string
    print "####### predict done ######## "

    print "####### print result ######## "
    st, resp = model_tool.print_predict_result(returned_state_string, correct_state_string, origin_sentence)
    if not st:
        print resp
    print "####### TEST END ######## "


# 使用算法训练模型之后的数据
def test_case_3():
    pass


def test_xiaoming():
    # 1 测试训练好的模型
    input_model = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\cppjieba-master\\dict\\hmm_model.utf8"
    input_model = "/Users/kylin/Downloads/AI LAB2/cppjieba-master/dict/hmm_model.utf8"
    sentence = u"小明硕士毕业于中国科学院计算所"
    correct_states_string = "BEBEBIEBEBIEBES"
    test_prediction(input_model, sentence, correct_states_string)


def real_test():
    # 2 训练并测试模型
    # 训练集路径
    # training_data = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\train_10_a"
    # training_data = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\train_10"
    training_data = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\train.utf8"
    training_data = "/Users/kylin/Downloads/AI LAB2/train.utf8"

    # 训练集的输出model的路径
    output_model = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\lab_train_data.dat"
    output_model = "/Users/kylin/Downloads/output_model"

    # 测试集路径
    # test_data_path = "E:\\download\\WeChat Files\\moonkylin14\\Files\\AI LAB2\\test_file_a"
    test_data_path = training_data

    test_train_and_predict(training_data, output_model, test_data_path)

if __name__ == "__main__":
    real_test()

