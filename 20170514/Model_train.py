import scipy as sp
import pandas as pd
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier


def file_read(filename_train, filename_test):
    """
    读取csv文件，转换为pandas DataFrame。
    :param filename_train: 训练集的csv文件名称
    :param filename_test: 测试集的csv文件名称
    :return: 训练集, 验证集, 测试集
    """
    total_set = pd.read_csv(filename_train)
    divide_point = int(len(total_set) * 0.8)
    train_set = total_set.head(divide_point)
    validation_set = total_set[divide_point:]

    test_set = pd.read_csv(filename_test)
    #print(type(train_set))
    #print(len(train_set))
    #print(train_set.head(5))
    #print(len(validation_set))
    #print(validation_set.head(5))
    return train_set, validation_set, test_set


def model_build(train_set):
    X = train_set.iloc[:, 6:11]
    Y = train_set['label']
    #print(X.head(5))
    #print(Y.head(5))
    model = GradientBoostingRegressor()
    #model = GradientBoostingClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    #print(model)
    return model


def model_prediction(model, validation_set=None, test_set=None):
    #X = validation_set.iloc[:, 6:11]
    X = test_set.iloc[:, 6:11]
    prediction = model.predict(X)
    #print(prediction)
    #print(X.head(5))
    return prediction


def file_output(label, prediction):
    instanceID = test_set.iloc[:, 0] + 1
    #result = pd.DataFrame({'instanceID': instanceID, 'label': label, 'prob': prediction.astype(np.double)})
    result = pd.DataFrame({'instanceID': instanceID, 'prob': prediction.astype(np.double)})
    result.to_csv("result.csv", index=False)


def logloss(label, prediction):
    epsilon = 1e-15
    prediction = sp.maximum(epsilon, prediction)
    prediction = sp.minimum(1-epsilon, prediction)
    ll = sum(label*sp.log(prediction) + sp.subtract(1,label)*sp.log(sp.subtract(1,prediction)))
    ll = ll * -1.0/len(label)
    print(ll)


if __name__ == "__main__":
    train_set, validation_set, test_set = file_read("output_set.csv", "output_set_test.csv")
    model = model_build(train_set)
    #prediction = model_prediction(model, validation_set)
    prediction = model_prediction(model, test_set=test_set)
    #label = validation_set.iloc[:, 1]
    #logloss(label, prediction)
    label = test_set.iloc[:, 1]
    file_output(label, prediction)
