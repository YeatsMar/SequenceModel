# coding=utf-8


# 把特征函数和对应的权重写入文件
def write_model_to_file(feature_func_obj_list, func_param_vector, model_output_file):
    model_output_ptr = open(model_output_file, "w+")
    for i in range(len(feature_func_obj_list)):
        func_obj = feature_func_obj_list[i]
        param_for_func = func_param_vector[i]
        model_output_ptr.write(str(i) + "\t" + str(param_for_func) + '\t' + func_obj.to_string() + '\n')
    model_output_ptr.close()
    return True, ''


# 只输出对应权重为正的特征函数
def write_positive_func_to_file(feature_func_obj_list, func_param_vector, model_output_file):
    model_output_ptr = open(model_output_file, "w+")
    for i in range(len(feature_func_obj_list)):
        func_obj = feature_func_obj_list[i]
        param_for_func = func_param_vector[i]
        if param_for_func > 0:
            model_output_ptr.write(str(i) + "\t" + str(param_for_func) + '\t' + func_obj.to_postitive_string() + '\n')
    model_output_ptr.close()
    return True, ''
