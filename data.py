from ucimlrepo import fetch_ucirepo
import pandas as pd
from dataanalysis import *
from datamodelling import *
  
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
#print(predict_students_dropout_and_academic_success.metadata) 
  
# variable information 
#print(predict_students_dropout_and_academic_success.variables)
description=predict_students_dropout_and_academic_success.variables
"""-----------------------------------------------------------------------------------------------------------------------"""
#READ DATA SET

Dataset=pd.concat([X,y],axis=1)

#print(Dataset.head(10))


"""-----------------------------------------------------------------------------------------------------------------------"""

#DATA ANALYSIS

Ana_Data, org_Data=DataAnalysis(Dataset)

#DATA MODELLING

Predict_data=DataModelling(Ana_Data, org_Data)


