import scipy as sp
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import FileOperator
import DataSeparate


def model_build(train_set, validation_set, test_set):
    X = train_set.iloc[:, 4:27]
    Y = train_set['label']
    dtrain = xgb.DMatrix(X, label=Y)
    print(dtrain.feature_names)
    #print(dtrain.get_label())
    #param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'reg:linear', 'eval_metric': 'logloss'}
    watch_list = [(dtrain, 'train')]
    param = {
            'nthread': 2,
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': 0.5,
            'max_depth': 12,
            'subsample': 1.0,
            'eval_metric': 'logloss'
    }
    bst = xgb.train(param, dtrain=dtrain, evals=watch_list)
    #prediction = bst.predict(X)
    return bst


def model_prediction(model, validation_set=None, test_set=None):
    #X = validation_set.iloc[:, 4:27]
    X = test_set.iloc[:, 4:27]
    dvalidation = xgb.DMatrix(X)
    prediction = model.predict(dvalidation)
    return  prediction


def feature_importance(model):
    ax = xgb.plot_importance(model)
    ax.plot()
    plt.show()


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


def change_prediction(prediction):
    #sort_prediction = prediction.sort_values(by='prediction', ascending=False)
    sort_prediction = -np.sort(-prediction)
    print(prediction)
    separate_point = int(0.9 * len(sort_prediction))
    separate_prediction = sort_prediction[separate_point]
    for i in range(len(prediction)):
        if prediction[i] < separate_prediction:
            prediction[i] *= 0.25
    print(prediction)
    return prediction


if __name__ == "__main__":
    train_set, validation_set, test_set = FileOperator.file_read("output_set.csv", "output_set_test.csv")
    bst = model_build(train_set, validation_set, test_set)
    #prediction = model_prediction(bst, validation_set=validation_set)
    prediction = model_prediction(bst, test_set=test_set)
    #train_set_list = DataSeparate.train_set_separate(train_set)
    #validation_set = train_set_list[0]
    #total_prediction = model_process(train_set_list, validation_set, test_set)
    #parameter_choose(train_set_list[0])
    #model_process([train_set], validation_set, test_set)
    #model = model_build(train_set)
    #prediction = model_prediction(model, validation_set)
    #prediction = model_prediction(model, test_set=test_set)
    #label = validation_set.iloc[:, 1]
    #logloss(label, prediction)
    #print(validation_set.head(5))
    #prediction = change_prediction(prediction)
    #logloss(label, prediction)
    #feature_importance(bst)
    #instanceID = validation_set.iloc[:, 0] + 1
    #FileOperator.file_output(instanceID, prediction, label=label)
    instanceID = test_set.iloc[:, 0] + 1
    FileOperator.file_output(instanceID, prediction)