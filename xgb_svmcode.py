import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,auc
import xgboost as xgb
from sklearn.svm import SVC
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score, f1_score, precision_score, \
    accuracy_score, confusion_matrix,f1_score
def readtrain_0():
    filepatheq = "Train_0_36_1.txt"
    filepatheq1 = "Train_0_36_2.txt"
    f = open(filepatheq, "r")
    eqlist = []
    for line in f.readlines():
        all_data = line.split()
        eqlist.append(all_data)
    eqlist = np.array(eqlist)
    eqlist = eqlist[:,3:]

    eqlist = np.float64(eqlist)
    data_eq = pd.DataFrame(eqlist)
    data_eq.insert(0, column='y', value=0)
    # 训练集
    f = open(filepatheq1, "r")
    eqlist = []
    for line in f.readlines():
        all_data = line.split()
        eqlist.append(all_data)
    eqlist = np.array(eqlist)
    eqlist = eqlist[:,3:]
    eqlist = np.float64(eqlist)
    data_ex = pd.DataFrame(eqlist)
    data_ex.insert(0, column='y', value=0)
    train_data = np.vstack((data_eq.values, data_ex.values))
    train_data_0 = pd.DataFrame(train_data)
    return train_data_0


def readtrain_1():
    filepatheq = "Train_1_36_1.txt"
    filepatheq1 = "Train_1_36_2.txt"
    f = open(filepatheq, "r")
    eqlist = []
    for line in f.readlines():
        all_data = line.split()
        eqlist.append(all_data)
    eqlist = np.array(eqlist)
    eqlist = eqlist[:, 3:]

    eqlist = np.float64(eqlist)
    data_eq = pd.DataFrame(eqlist)
    data_eq.insert(0, column='y', value=1)
    # 训练集
    f = open(filepatheq1, "r")
    eqlist = []
    for line in f.readlines():
        all_data = line.split()
        eqlist.append(all_data)
    eqlist = np.array(eqlist)
    eqlist = eqlist[:, 3:]
    eqlist = np.float64(eqlist)
    data_ex = pd.DataFrame(eqlist)
    data_ex.insert(0, column='y', value=1)
    train_data = np.vstack((data_eq.values, data_ex.values))
    train_data_0 = pd.DataFrame(train_data)
    return train_data_0

def readtest2_01():
    filepatheq = "Test2_0_36.txt"
    filepatheq1 = "Test2_1_36.txt"
    f = open(filepatheq, "r")
    eqlist = []
    for line in f.readlines():
        all_data = line.split()
        eqlist.append(all_data)
    eqlist = np.array(eqlist)
    eqlist = eqlist[:, 3:]

    eqlist = np.float64(eqlist)
    data_eq = pd.DataFrame(eqlist)
    data_eq.insert(0, column='y', value=0)
    # 训练集
    f = open(filepatheq1, "r")
    eqlist = []
    for line in f.readlines():
        all_data = line.split()
        eqlist.append(all_data)
    eqlist = np.array(eqlist)
    eqlist = eqlist[:, 3:]
    eqlist = np.float64(eqlist)
    data_ex = pd.DataFrame(eqlist)
    data_ex.insert(0, column='y', value=1)
    train_data = np.vstack((data_eq.values, data_ex.values))
    train_data_0 = pd.DataFrame(train_data)
    return train_data_0
def dataprocess():

    df_0 =  readtrain_0()
    df_1 =  readtrain_1()
    df_test2 = readtest2_01()
    df_train = np.vstack((df_0.values, df_1.values))
    df_train = pd.DataFrame(df_train)

    df_train.columns = range(df_train.shape[1])
    # 某列数据类型转化
    df_train[0] = df_train[0].astype(int)
    data_eq = df_train[df_train[0] == 0]
    data_ex = df_train[df_train[0] == 1]
    num = len(data_eq)
    num1 = len(data_ex)
    if(num>num1):
        num =num1
    print(num)
    data_eq = df_train[df_train[0] == 0].sample(num)
    data_ex = df_train[df_train[0] == 1].sample(num)
    train_data = np.vstack((data_eq.values, data_ex.values))
    train_data = pd.DataFrame(train_data)
    trainX = train_data.iloc[:,1:].values
    trainY = train_data.iloc[:,0].values

    df_test2.columns = range(df_test2.shape[1])
    # 某列数据类型转化
    df_test2[0] = df_test2[0].astype(int)

    x_test2 = df_test2.iloc[:,1:].values
    y_test2 = df_test2.iloc[:,0].values
    x_train, x_test1, y_train, y_test1 = train_test_split(trainX, trainY, test_size=0.1, random_state=None, stratify=trainY)

    pipe_lrpro = Pipeline([('sc', MinMaxScaler())])
    pipe_lrpro.fit(x_train)
    x_train = pipe_lrpro.transform(x_train)
    x_test1 = pipe_lrpro.transform(x_test1)
    x_test2 = pipe_lrpro.transform(x_test2)

    return x_train,x_test1,x_test2,y_train,y_test1,y_test2

def xgboostkfold(X_train, y_train, x_vali, y_vali,x_test1,y_test1, x_test2, y_test2 ):
    resultlist = []
    start = time.perf_counter()

  #  model = SVC(C=1, gamma=1, kernel='rbf')
    # svm = SVC(C=100, gamma=0.01, kernel='rbf')
    model = xgb.XGBClassifier(
        # learning_rate=0.1,
        n_estimators=500,
        max_depth=10
        # max_depth=3,
        # learning_rate=0.01,
        # n_estimators=60
    )
    model.fit(X_train, y_train)
    end = time.perf_counter()
    runtime = end - start

    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    y_pred_test1 = model.predict(x_test1)
    y_pred_vali = model.predict(x_vali)




    vali_acc = accuracy_score(y_vali, y_pred_vali)
    vali_recall = recall_score(y_vali, y_pred_vali, average=None)
    vali_f1 = f1_score(y_vali, y_pred_vali, average='weighted')
    # 测试集
    df = pd.DataFrame(confusion_matrix(y_vali, y_pred_vali))
    vali_recall_num = list()
    vali_recall_num.append(df.iloc[0, 0])
    vali_recall_num.append(df.iloc[1, 1])

    aucscore = 1
    aucscore_vali = 1
    roc_test = []


    svm_test_probs = model.decision_function(x_vali)
    lr_fpr_test, lr_tpr_test, _ = roc_curve(y_vali, svm_test_probs)
    aucscore = auc(lr_fpr_test, lr_tpr_test)
    fpr = [float('{:.6f}'.format(i)) for i in list(lr_fpr_test)]
    tpr = [float('{:.6f}'.format(i)) for i in list(lr_tpr_test)]
    roc_test.append(fpr)
    roc_test.append(tpr)

    resultlist.append(runtime)
    resultlist.append(vali_acc)
    resultlist.append(aucscore)
    resultlist.append(vali_f1)
    resultlist.append(vali_recall[0])
    resultlist.append(vali_recall[1])

    return resultlist ,roc_test

def SVMTrain():

    df5 = pd.DataFrame(columns=['run_time','SVM_acc_vali', 'SVM_auc_vali', 'SVM_f1_vali', 'SVM_acc0_vali', 'SVM_acc1_vali'],
                       index=[])
    df6 = pd.DataFrame(columns=['SVM_acc_test1','SVM_f1_test1', 'SVM_acc0_test1', 'SVM_acc1_test1', 'SVM_num0_test1', 'SVM_num1_test1'],
                        index=[])
    df7 = pd.DataFrame(columns=['SVM_acc_test2','SVM_f1_test2', 'SVM_acc0_test2', 'SVM_acc1_test2', 'SVM_num0_test2', 'SVM_num1_test2'],
                        index=[])

    dfxgb = pd.DataFrame()
    x_train,x_test1,x_test2,y_train,y_test1,y_test2 = dataprocess()
    num0 = (y_train == 0).sum()
    num1 = (y_train == 1).sum()
    num2 = (y_test1 == 0).sum()
    num3 = (y_test1 == 1).sum()
    cv = KFold(n_splits=10)
    roc_save = []

    for train_index, test_index in cv.split(x_train):
        X_train, X_vali, Y_train, Y_vali = x_train[train_index], x_train[test_index], y_train[train_index], y_train[test_index]
        xgbresult,roc = xgboostkfold(X_train,Y_train,X_vali,Y_vali,x_test1,y_test1,x_test2,y_test2)


if __name__ == '__main__':

    SVMTrain()

