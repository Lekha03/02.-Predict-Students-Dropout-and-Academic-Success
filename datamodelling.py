import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def DataModelling(dataset, org_data):

    #Logistic_regression
    Y_log=dataset[['Target_Enrolled']]
    X_log=dataset.drop(['Target_Enrolled'], axis=1)
    X_train_log,X_test_log,Y_train_log, Y_test_log=train_test_split(X_log,Y_log,test_size=0.3,random_state=1234,stratify=Y)
    lr=LogisticRegression()
    lr.fit(X_train_log, Y_train_log)
    Y_predict_log=lr.predict(X_test_log)
    
    cm_log=confusion_matrix(Y_test_log, Y_predict_log)
    score_log=lr.score(X_test_log, Y_test_log)

    print(cm_log)
    print(score_log)
    print(Y_predict_log)



    #Build SVM

    Y_svm=dataset[['Target_Enrolled']]
    X_svm=dataset.drop(['Target_Enrolled'], axis=1)
    X_train_svm,X_test_svm,Y_train_svm, Y_test_svm=train_test_split(X_svm,Y_svm,test_size=0.3,random_state=1234,stratify=Y)

    from sklearn.svm import SVC

    #predict the outcome of the data
    svc=SVC(kernel='linear',gamma=0.1)
    svc.fit(X_train_svm, Y_train_svm)
    y_predict_svc=svc.predict(X_test_svm)
    cm_svc=confusion_matrix(Y_test_svm, y_predict_svc)
    score_svc=svc.score(X_test_svm, Y_test_svm)
    print(cm_svc)
    print(score_svc)
    print(y_predict_svc)


    #






