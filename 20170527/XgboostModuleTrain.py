import scipy as sp
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import FileOperator

'''
def model_build(train_set):
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
'''


def model_build(train_set):
    X = train_set.iloc[:500000, 4:27]
    Y = train_set.iloc[:500000, 1]
    #Y = train_set['label']
    dtrain = xgb.DMatrix(X, label=Y)
    model = XGBClassifier(learning_rate=0.5,
                          n_estimators=20,
                          max_depth=7,
                          min_child_weight=5,
                          gamma=0.4,
                          subsample=1,
                          objective='binary:logistic',
                          nthread=4,
                          scale_pos_weight=1,
                          colsample_bytree=0.2,
                          reg_alpha=1)
    print(model)
    #cross_validation(model, dtrain)
    grid_search(model, X, Y)
    model.fit(X, Y)
    return model


def model_prediction(model, predict_set):
    #X = validation_set.iloc[:, 4:27]
    X = predict_set.iloc[:, 4:27]
    #dvalidation = xgb.DMatrix(X)
    prediction = model.predict_proba(X)[:, 1]
    return prediction


def cross_validation(model, dtrain, cv_flods=5, early_stopping_rounds=50):
    xgb_param = model.get_xgb_params()
    print(xgb_param)
    cv_result = xgb.cv(xgb_param,
                       dtrain,
                       num_boost_round=xgb_param.get('n_estimators'),
                       nfold=cv_flods,
                       metrics='logloss',
                       early_stopping_rounds=early_stopping_rounds,
                       show_stdv=True)
    model.set_params(n_estimators=cv_result.shape[0])
    print(model.get_xgb_params()['n_estimators'])


def grid_search(model, X, Y):
    param_test1 = {'max_depth': range(5, 15, 2), 'min_child_weight': range(1, 6, 2)}
    param_test2 = {'gamma': [i/10.0 for i in range(0, 5, 1)]}
    param_test3 = {'subsample': [i/10.0 for i in range(10, 11)], 'colsample_bytree': [i/10.0 for i in range(1, 5)]}
    param_test4 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(estimator=model,
                           param_grid=param_test4,
                           scoring='neg_log_loss',
                           cv=5)
    gsearch.fit(X, Y)
    print('params:', gsearch.cv_results_['params'])
    print('mean_test_score:', gsearch.cv_results_['mean_test_score'])
    print('std_test_score:', gsearch.cv_results_['std_test_score'])

    print(gsearch.best_estimator_)
    print(gsearch.best_params_)


def feature_importance(model):
    #fscore = model.get_fscore()
    #print(fscore)
    #print(type(fscore))
    #feat_imp = pd.Series(fscore).sort_values(ascending=False)
    #ax = feat_imp.plot(kind='bar', title='Feature Importance')
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
    model = model_build(train_set)
    #bst = model_build(train_set, validation_set, test_set)
    #prediction = model_prediction(model, validation_set)

    prediction = model_prediction(model, test_set)
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
    feature_importance(model)
    instanceID = test_set.iloc[:, 0] + 1
    FileOperator.file_output(instanceID, prediction)
