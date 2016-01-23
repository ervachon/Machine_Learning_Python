# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:39:09 2016

@author: eriva
"""
import pandas as pd
import numpy  as np
import os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from ggplot import *

DataDir           = 'data'
DataFileTraining  = 'pml-training.csv'
DataFileTest      = 'pml-testing.csv'
 
os.chdir ("D:\_GIT_\Machine_Learning_Python\Practical_Machine_Learning_Project")
completePathTrain = os.path.join(os.getcwd(),DataDir,DataFileTraining)
print("> import training data : " +  completePathTrain)
myDF = pd.read_csv(completePathTrain, sep=',',low_memory=False)
myDF = myDF.reindex(np.random.permutation(myDF.index)).reset_index(drop=True)

completePathTest = os.path.join(os.getcwd(),DataDir,DataFileTest)
print("> import test data : " +  completePathTest )
myDFTest = pd.read_csv(completePathTest , sep=',',low_memory=False)

#ggplot
ggplot(aes(x = 'classe'),data=myDF) + geom_histogram()

#list of column who  has name 'timestamp' 
lTimeStamp =  myDF.filter(like='timestamp', axis=1).columns
myDF = myDF.drop(lTimeStamp,axis=1)
myDFTest = myDFTest.drop(lTimeStamp,axis=1)

myDF = myDF.drop("Unnamed: 0",axis=1)
myDFTest = myDFTest.drop("Unnamed: 0",axis=1)

myDF["user_name"].isnull().sum()
countNull = pd.Series([100*myDF[col].isnull().sum()/len(myDF) for col in myDF.columns], index = myDF.columns)
colsNullDrop = countNull[countNull >= 0.95].index.tolist()
myDF = myDF.drop(colsNullDrop,axis=1)
myDFTest = myDFTest.drop(colsNullDrop,axis=1)

myLabelEncoder = preprocessing.LabelEncoder()
myLabelEncoder.fit(myDF["new_window"])
myDF["new_window"] = myLabelEncoder.transform(myDF["new_window"])
myDFTest["new_window"] =  myLabelEncoder.transform(myDFTest["new_window"])

myLabelEncoder = preprocessing.LabelEncoder()
myLabelEncoder.fit(myDF["user_name"])
myDF["user_name"] = myLabelEncoder.transform(myDF["user_name"])
myDFTest["user_name"] =  myLabelEncoder.transform(myDFTest["user_name"])




       
# myDFY = ((myDF.CHURN_INV_M4 == 1) | (myDF.TRT_Code_Decision==2)).astype(int)
# myDFX = myDF[listKeyColums]
# print("> add cols ok")
# myDFX[listNumericData] = myDF[listNumericData]

# print("> filter columns that contains more than 100 unique values")
# nunique = pd.Series([myDFX[col].nunique() for col in myDFX.columns], index = myDFX.columns)
# cols = nunique[nunique < 100].index.tolist()
# myDFX = myDFX[cols]
    
# convert to numbefor aColumn in listCategorielData:r the labels 
# for aColumn in listCategorielData:
# myDFX[aColumn] = transformLabelIntoInteger(myDF[aColumn])
    
#print("> drop constants columns")
#nunique = pd.Series([myDFX[col].nunique() for col in myDFX.columns], index = myDFX.columns)
#cols = nunique[nunique == 1].index.tolist()
#myDFX = myDFX.drop(cols,axis=1)

#print("> create test / train")
# 90% train / 10% test
#myDFX['is_train'] = np.random.uniform(0, 1, len(myDFX)) <= .9 
#myTrainX, myTestX = myDFX[myDFX['is_train']==True], myDFX[myDFX['is_train']==False]
#myTrainY, myTestY = myDFY[myDFX['is_train']==True], myDFY[myDFX['is_train']==False]

#print("> RF")
# now RF    
#myForest = RandomForestClassifier(n_estimators = 100,n_jobs=4)
#myForest.fit(myTrainX,myTrainY)

# myPreds = iris.target_names[clf.predict(test[features])]
# pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])
