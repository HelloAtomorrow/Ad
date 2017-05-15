import pandas as pd
import numpy as np


def file_read(filename_train, filename_test=None):
    """
    读取csv文件，转换为pandas DataFrame。
    :param filename_train: 训练集的csv文件名称
    :param filename_test: 测试集的csv文件名称
    :return: 训练集, 验证集, 测试集
    """
    total_set = pd.read_csv(filename_train)
    sample_num = int(len(total_set) * 0.8)
    random_set = np.random.permutation(len(total_set))
    train_set = total_set.take(random_set[:sample_num])
    #print(train_set['label'].value_counts())
    validation_set = total_set.take(random_set[sample_num:])
    #print(validation_set['label'].value_counts())
    if not filename_test:
        test_set = None
    else:
        test_set = pd.read_csv(filename_test)
    #print(type(train_set))
    #print(len(train_set))
    #print(train_set.head(5))
    #print(len(validation_set))
    #print(validation_set.head(5))
    return train_set, validation_set, test_set


def file_output(instanceID, prediction, label=None):
    """
    将预测结果输出为csv文件，根据label是否为空决定输出验证集还是测试集
    :param instanceID: 预测编号
    :param prediction: 预测回流概率
    :param label: 实际回流状态
    :return: 无
    """
    #instanceID = test_set.iloc[:, 0] + 1
    if not isinstance(label, pd.core.series.Series):
        result = pd.DataFrame({'instanceID': instanceID, 'prob': prediction.astype(np.double)})
    else:
        result = pd.DataFrame({'instanceID': instanceID, 'label': label, 'prob': prediction.astype(np.double)})
    result.to_csv("result.csv", index=False)


if __name__ == "__main__":
    print("file operator")