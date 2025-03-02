import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def DataAnalysis(dataset):
    #print(dataset.columns)

    #DATA CLEANING
    #print(dataset.isnull().sum(axis=0))
    # No Missing Values
    # 36 columns
    
    #print(dataset)
    #DATA TRANSFORMATION
    print(dataset.dtypes)
    #according to the features and their description some of the columns are integers but are not ocntinuous and indictae some value.
    #Update those columns to catego objectry dtype and create dummies
    categorical_features=['Marital Status', 'Application mode','Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification','Nacionality', "Mother's qualification", "Father's qualification","Mother's occupation","Father's occupation","Displaced", "Educational special needs","Debtor","Tuition fees up to date","Gender","Scholarship holder","International"]
    for i in categorical_features:
        dataset[i]=dataset[i].astype('object')
    #print(dataset.dtypes)
    #dataset=pd.get_dummies(dataset,drop_first=True)
    
    #org_data=dataset.copy()
    #DATA NORMALIZATION
    print(dataset.head(5))
    normal=[]
    scaler_=StandardScaler()
    for i in dataset.columns:
        if dataset[i].dtypes == 'int64' or dataset[i].dtypes == 'float64':
            dataset[i]=scaler_.fit_transform(dataset[[i]])
            normal.append(i)
    print(dataset.head(5))
    print(normal)


    return dataset




