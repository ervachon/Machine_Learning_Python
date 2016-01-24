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
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, classification_report
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score
from sklearn.cross_validation import train_test_split
import re

def returnDomain(email):
    strTmp = ""
    searchDomain = re.search( r'^(.*)@(.*)$', email, re.M|re.I)
    if searchDomain:
       #print "searchDomain.group() : ", searchDomain.group()
       #print "searchDomain.group(1) : ", searchDomain.group(1)
       #print "searchDomain.group(2) : ", searchDomain.group(2)
       strTmp = searchDomain.group(2)
    else:
       strTmp = "NA"
    return(strTmp)
    
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


#print("> drop constants columns")
#nunique = pd.Series([myDFX[col].nunique() for col in myDFX.columns], index = myDFX.columns)
#cols = nunique[nunique == 1].index.tolist()
#myDFX = myDFX.drop(cols,axis=1)

# print("> filter columns that contains more than 100 unique values")
# nunique = pd.Series([myDFX[col].nunique() for col in myDFX.columns], index = myDFX.columns)
# cols = nunique[nunique < 100].index.tolist()
# myDFX = myDFX[cols]
      
# now 80% / 20%
X_train, X_test, y_train, y_test = train_test_split(myDF.ix[:,0:-1], myDF.ix[:,-1], test_size=0.2)

myForest = RandomForestClassifier(n_estimators = 100,n_jobs=3)
myForest.fit(X_train,y_train)

y_testHat = myForest.predict(X_test)
confusionMatrix = confusion_matrix(y_test, y_testHat, labels=None)

# faire le y_test en multi colonnes
y_testHatProba = myForest.predict_proba(X_test)

listColumns = np.sort(y_train.unique())

y_testProba = pd.DataFrame(y_test)
for col in listColumns:
    y_testProba[col]=(y_testProba['classe'] == col)*1
y_testProba = y_testProba.drop('classe',axis=1)

roc_auc = roc_auc_score(y_testProba, y_testHatProba)

###############################################################################
accScore  = accuracy_score(y_test, y_testHat)
f1_score  = f1_score(y_test, y_testHat,average='weighted')
recall    = recall_score(y_test, y_testHat, average='weighted')
precision = precision_score(y_test, y_testHat,average='weighted')
report    = classification_report(y_test, y_testHat)
###############################################################################

# for plot
completRes = pd.DataFrame(y_test)
completRes['Prediction'] = y_testHat
completRes.columns = ['Reference', 'Prediction'] 

# jitter ????
# corrplot(cor(trainDummy[,-idxClasse]), order = "FPC", method = "color", type = "lower", tl.cex = 0.6,title="\n\nCorrelation between features" )
# plot(varImp(modelFit), top = 40)

###############################################################################



