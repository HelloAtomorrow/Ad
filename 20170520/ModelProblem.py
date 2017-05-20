import scipy as sp
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model.logistic import logistic_regression_path
import random
import pandas as pd


def data_prepare():
    NUM = 100000
    label1 = [1] * int(0.25 * NUM)
    #print(label1)
    label2 = [0] * int(0.75 * NUM)
    #print(label2)
    #label = label1.extend(label2)
    label = label1 + label2
    #print(len(label))

    age1 = [random.randint(40, 50) for i in range(int(0.25 * NUM))]
    age2 = [random.randint(10, 40) for i in range(int(0.75 * NUM))]
    #print(age1)
    #print(age2)
    age = age1 + age2

    marriage1 = [1] * int(0.25 * NUM)
    marriage2 = [1] * int(0.75 * NUM)
    marriage = marriage1 + marriage2

    train_set = pd.DataFrame({'label': label, 'age': age, 'marriage': marriage})
    #print(train_set.head(5))
    return train_set


def model_build(train_set, weight=None):
    """
    模型建立，根据训练集，构建GBDT模型
    :param train_set: 训练集
    :param weight: 训练集label权重列表
    :return: 训练完成的model
    """
    X = train_set.iloc[:, 1:]
    print(len(X))
    Y = train_set['label']
    print(len(Y))
    #print(X.head(5))
    #print(Y.head(5))
    model = GradientBoostingRegressor()
    #model = GradientBoostingClassifier()
    #model = logistic_regression_path(X, Y)
    model.fit(X, Y)
    print(model.feature_importances_)
    #print(model)
    return model


def model_prediction(model, validation_set=None, test_set=None):
    """
    模型预测，使用已经训练好的模型对验证集或者测试集预测
    :param model: 训练后的GBDT模型
    :param validation_set: 验证集
    :param test_set: 测试集
    :return: 预测结果
    """
    X = train_set.iloc[:, 1:]
    #X = test_set.iloc[:, 6:11]
    prediction = model.predict(X)
    print(prediction)
    #print(X.head(5))
    return prediction


def logloss(label, prediction):
    """
    计算损失函数，对验证集的验证结果与实际label做比较
    :param label: 实际标签值
    :param prediction: 预测值
    :return: 对数损失logloss
    """
    epsilon = 1e-15
    prediction = sp.maximum(epsilon, prediction)
    prediction = sp.minimum(1-epsilon, prediction)
    ll = sum(label*sp.log(prediction) + sp.subtract(1,label)*sp.log(sp.subtract(1,prediction)))
    ll = ll * -1.0/len(label)
    print(ll)
    return ll


if __name__ == '__main__':
    train_set = data_prepare()
    model = model_build(train_set)
    prediction = model_prediction(model, validation_set=train_set)
    logloss(train_set['label'], prediction)