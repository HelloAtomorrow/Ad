import FileOperator
import pandas as pd


def train_set_separate(train_set_origin):
    """
    将训练集划分为多个，划分后的每个训练集包含全部label为1的样本以及部分label为0的样本
    :param train_set_origin: 需要划分的原数据集
    :return: 划分后的数据集列表
    """
    train_set_sort = train_set_origin.sort_values(by='label', ascending=False)
    #tt = train_set_sort['label'].value_counts()
    backflow_num = train_set_sort['label'].sum()
    notflow_num = len(train_set_sort) - backflow_num
    print("回流样本数量：", backflow_num)
    backflow_set = train_set_sort.iloc[0:backflow_num, :]
    #print(backflow_set.head(5))

    notflow_set = train_set_sort.iloc[backflow_num:, :]
    #print(notflow_set.head(5))
    train_set_list = []

    for i in range(10):
        #sample_num = int(notflow_num / 10)
        #notflow_set_sample = notflow_set.take(np.random.permutation(len(notflow_set))[:sample_num])
        notflow_set_sample = notflow_set.sample(frac=0.1)
        train_set_sample = pd.concat([backflow_set, notflow_set_sample])
        #print(len(train_set_sample))
        train_set_list.append(train_set_sample)
    #print(train_set_list[0].head(5))
    return train_set_list


if __name__ == "__main__":
    train_set, validation_set, test_set = FileOperator.file_read("output_set.csv")
    train_set_list = train_set_separate(train_set)


