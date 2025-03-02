import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
#from sklearn import cross_validation

def DataModelling(dataset):

    #Variables
    Y=dataset[['Target']]
    X=dataset.drop(['Target'], axis=1)
    Y=pd.get_dummies(Y,drop_first=True)
    X=pd.get_dummies(X,drop_first=True)
    print(Y)
    X_train_log,X_test_log,Y_train_log, Y_test_log=train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)


    '''#Logistic regression
    lr=LogisticRegression()
    lr.fit(X_train_log, Y_train_log)
    Y_predict_log=lr.predict(X_test_log)
    
    cm_log=confusion_matrix(Y_test_log, Y_predict_log)
    score_log=lr.score(X_test_log, Y_test_log)

    print(cm_log)
    print(score_log)
    print(Y_predict_log)'''



    #Build SVM

    # X_train_svm,X_test_svm,Y_train_svm, Y_test_svm=train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)

    # from sklearn.svm import SVC
    # print("SVM")
    # #predict the outcome of the data
    # svc=SVC(kernel='linear',gamma=0.1)
    # svc.fit(X_train_svm, Y_train_svm)
    # y_predict_svc=svc.predict(X_test_svm)
    # cm_svc=confusion_matrix(Y_test_svm, y_predict_svc)
    # score_svc=svc.score(X_test_svm, Y_test_svm)
    # print("Confusion_mtix",cm_svc)
    # print("Score",score_svc)
    # print("y_predict_svc",y_predict_svc)


    #RFC
    from sklearn.ensemble import RandomForestClassifier

    rfc1=RandomForestClassifier(random_state=1234)
    rfc1.fit(X_train_log, Y_train_log)

    y_predict=rfc1.predict(X_test_log)

    cm1=confusion_matrix(Y_test_log.values.argmax(axis=1), y_predict.argmax(axis=1))

    score1=rfc1.score(X_test_log, Y_test_log)

    print(cm1)
    print(score1)

    print(X.shape)

    from sklearn.feature_selection import f_classif as fc

    result=fc(X,Y.values.argmax(axis=1))
    f_score=result[0]
    p_values=result[1]

    f_class_data=pd.concat([pd.DataFrame(result).transpose(),pd.DataFrame(X.columns)],axis=1)
    #f_class_data.to_excel('feature.xlsx')
    undata=[]
    print(f_class_data)
    for _,i in f_class_data.iterrows():
        if i[1]>0.1:
            undata.append(f"{i.iloc[2]}")
    #print(undata)
    newdataset=dataset.copy()
    #for i in undata:
    import re
    for col in undata:
    # Find matching column in df.columns
        pattern = re.escape(col)
        matching_col = next((c for c in newdataset.columns if re.fullmatch(pattern, c)), None)
        if matching_col:
            newdataset.drop(columns=[matching_col], inplace=True)
    #print(newdataset)
    #newdataset=newdataset.drop(['Curricular units 1st sem (evaluations)', 'Curricular units 2nd sem (evaluations)', 'Inflation rate', 'Marital Status_3', 'Marital Status_5', 'Marital Status_6', 'Application mode_2', 'Application mode_5', 'Application mode_10', 'Application mode_15', 'Application mode_17', 'Application mode_18', 'Application mode_26', 'Application mode_27', 'Application mode_42', 'Application mode_51', 'Application mode_53', 'Application mode_57', 'Application order_5', 'Application order_9', 'Course_171', 'Course_9085', 'Course_9254', 'Course_9556', 'Course_9670', 'Previous qualification_4', 'Previous qualification_5', 'Previous qualification_6', 'Previous qualification_10', 'Previous qualification_14', 'Previous qualification_15', 'Previous qualification_38', 'Previous qualification_40', 'Previous qualification_42', 'Previous qualification_43', 'Nacionality_2', 'Nacionality_6', 'Nacionality_13', 'Nacionality_14', 'Nacionality_17', 'Nacionality_21', 'Nacionality_22', 'Nacionality_24', 'Nacionality_25', 'Nacionality_26', 'Nacionality_32', 'Nacionality_41', 'Nacionality_62', 'Nacionality_101', 'Nacionality_103', 'Nacionality_105', 'Nacionality_108', 'Nacionality_109', "Mother's qualification_2", "Mother's qualification_4", "Mother's qualification_5", "Mother's qualification_6", "Mother's qualification_9", "Mother's qualification_10", "Mother's qualification_11", "Mother's qualification_12", "Mother's qualification_14", "Mother's qualification_18", "Mother's qualification_22", "Mother's qualification_26", "Mother's qualification_27", "Mother's qualification_29", "Mother's qualification_30", "Mother's qualification_35", "Mother's qualification_36", "Mother's qualification_39", "Mother's qualification_40", "Mother's qualification_41", "Mother's qualification_42", "Mother's qualification_43", "Mother's qualification_44", "Father's qualification_3", "Father's qualification_4", "Father's qualification_6", "Father's qualification_9", "Father's qualification_10", "Father's qualification_11", "Father's qualification_12", "Father's qualification_13", "Father's qualification_14", "Father's qualification_18", "Father's qualification_20", "Father's qualification_25", "Father's qualification_26", "Father's qualification_27", "Father's qualification_30", "Father's qualification_31", "Father's qualification_33", "Father's qualification_35", "Father's qualification_36", "Father's qualification_40", "Father's qualification_41", "Father's qualification_42", "Father's qualification_43", "Father's qualification_44", "Mother's occupation_1", "Mother's occupation_3", "Mother's occupation_4", "Mother's occupation_5", "Mother's occupation_6", "Mother's occupation_7", "Mother's occupation_8", "Mother's occupation_10", "Mother's occupation_122", "Mother's occupation_123", "Mother's occupation_125", "Mother's occupation_131", "Mother's occupation_132", "Mother's occupation_134", "Mother's occupation_141", "Mother's occupation_143", "Mother's occupation_144", "Mother's occupation_151", "Mother's occupation_152", "Mother's occupation_153", "Mother's occupation_171", "Mother's occupation_173", "Mother's occupation_175", "Mother's occupation_191", "Mother's occupation_192", "Mother's occupation_193", "Mother's occupation_194", "Father's occupation_1", "Father's occupation_3", "Father's occupation_5", "Father's occupation_6", "Father's occupation_8", "Father's occupation_9", "Father's occupation_10", "Father's occupation_101", "Father's occupation_102", "Father's occupation_103", "Father's occupation_112", "Father's occupation_114", "Father's occupation_121", "Father's occupation_122", "Father's occupation_123", "Father's occupation_124", "Father's occupation_131", "Father's occupation_132", "Father's occupation_134", "Father's occupation_135", "Father's occupation_141", "Father's occupation_143", "Father's occupation_144", "Father's occupation_151", "Father's occupation_153", "Father's occupation_154", "Father's occupation_161", "Father's occupation_163", "Father's occupation_171", "Father's occupation_172", "Father's occupation_174", "Father's occupation_175", "Father's occupation_181", "Father's occupation_182", "Father's occupation_183", "Father's occupation_192", "Father's occupation_193", "Father's occupation_194", "Father's occupation_195", 'Educational special needs_1', 'International_1'], axis=1)
    #newdataset=newdataset.drop(columns=undata,axis=1, inplace=True, errors='ignore')
    Y=newdataset[['Target']]
    X=newdataset.drop(['Target'], axis=1)
    Y=pd.get_dummies(Y,drop_first=True)
    X=pd.get_dummies(X,drop_first=True)
    #print(Y)
    X_train_log,X_test_log,Y_train_log, Y_test_log=train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)

    print("RandomForestClassifier random_state=1234")

    rfc2=RandomForestClassifier(random_state=1234)
    rfc2.fit(X_train_log, Y_train_log)
    y_predict=rfc2.predict(X_test_log)
    cm2=confusion_matrix(Y_test_log.values.argmax(axis=1), y_predict.argmax(axis=1))
    score2=rfc2.score(X_test_log, Y_test_log)
    print("confusion_matrix",cm2)
    print(score2)
    #print(X.shape)

    print("RandomForestClassifier - n_estimators=200,max_features=100, random_state=1234")
    rfc2=RandomForestClassifier(n_estimators=200,max_features=100, random_state=1234)
    rfe=RFE(estimator=rfc2)
    rfe.fit(X,Y)
    x_train_rfe=rfe.transform(X_train_log)
    x_test_rfe=rfe.transform(X_test_log)
    rfc2.fit(x_train_rfe,Y_train_log)
    y_predict=rfc2.predict(X_test_log)
    cm_rfe=confusion_matrix(Y_test_log.values.argmax(axis=1),y_predict.argmax(axis=1))
    score_rfe=rfc2.score(x_test_rfe,Y_test_log)
    print("confusion_matrix",cm_rfe)
    print("score",score_rfe)






