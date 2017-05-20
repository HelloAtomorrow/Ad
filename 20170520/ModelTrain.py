import scipy as sp
import pandas as pd
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import FileOperator
import DataSeparate


def model_process(train_set_list, validation_set, test_set):
    """
    对训练集列表中的模型依次训练，并使用验证集验证，找出效果最好的前几个，依次对测试集预测
    :param train_set_list: 训练集列表
    :param validation_set: 验证集
    :param test_set: 测试集
    :return: 最终的预测结果
    """
    model_list = []
    ll_list = []
    label = validation_set.iloc[:, 1]
    for train_set in train_set_list:
        model = model_build(train_set)
        prediction = model_prediction(model, validation_set)
        #prediction = model_prediction(model, train_set)###
        #label = train_set.iloc[:, 1]###
        model_list.append(model)
        ll = logloss(label, prediction)
        ll_list.append(ll)
        break

    instanceID = validation_set.iloc[:, 0] + 1
    #instanceID = train_set.iloc[:, 0] + 1
    FileOperator.file_output(instanceID, prediction, label=label)

    model_set = pd.DataFrame({'model': model_list, 'll': ll_list})
    model_set = model_set.sort_values(by='ll')
    #print(model_set)
    model_set = model_set.head(5)

    i = 0
    prediction_list = []
    for model in model_set['model']:
        i += 1
        joblib.dump(model, 'GBDT'+str(i)+'.model')
        prediction = model_prediction(model, test_set)
        prediction_list.append(prediction)

    total_prediction = 0.2 * prediction_list[0]
    for prediction in prediction_list[1:]:
        total_prediction += 0.2 * prediction
    return total_prediction


def parameter_choose(train_set):
    """
    模型最佳参数选择，根据对应的训练集选择最佳模型参数
    :param train_set: 训练集
    :return: 无
    """
    X = train_set.iloc[:, 6:11]
    Y = train_set['label']
    param_test = {'n_estimators': range(10, 81, 10)}
    gsearch = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=1), param_grid=param_test,
                           iid=True, cv=5)
    gsearch.fit(X, Y)
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.best_params_, gsearch.best_score_)


def model_build(train_set, weight=None):
    """
    模型建立，根据训练集，构建GBDT模型
    :param train_set: 训练集
    :param weight: 训练集label权重列表
    :return: 训练完成的model
    """
    X = train_set.iloc[:, 6:12]
    Y = train_set['label']
    #print(X.head(5))
    #print(Y.head(5))
    model = GradientBoostingRegressor()
    #model = GradientBoostingClassifier()
    if not weight:
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
    #X = validation_set.iloc[:, 6:12]
    X = test_set.iloc[:, 6:12]
    prediction = model.predict(X)
    #print(prediction)
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


if __name__ == "__main__":
    train_set, validation_set, test_set = FileOperator.file_read("pre_output_set.csv", "output_set_test.csv")
    train_set_list = DataSeparate.train_set_separate(train_set)
    #validation_set = train_set_list[0]
    #total_prediction = model_process(train_set_list, validation_set, test_set)
    #parameter_choose(train_set_list[0])
    #model_process([train_set], validation_set, test_set)
    model = model_build(train_set)
    #prediction = model_prediction(model, validation_set)
    prediction = model_prediction(model, test_set=test_set)
    #label = validation_set.iloc[:, 1]
    #logloss(label, prediction)
    #label = test_set.iloc[:, 1]
    instanceID = test_set.iloc[:, 0] + 1
    FileOperator.file_output(instanceID, prediction)
