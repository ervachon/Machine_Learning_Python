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
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

DataDir  = 'data'
DataFile = 'student-por.csv'
 
os.chdir ("D:\_GIT_\Machine_Learning_Python\Developing_Data_Products")
completePath = os.path.join(os.getcwd(),DataDir,DataFile)
print("> import data : " +  completePath)
myDF = pd.read_csv(completePath, sep=',',low_memory=False)
myDF = myDF.reindex(np.random.permutation(myDF.index)).reset_index(drop=True)


###############################################################################

theURL <-"https://archive.ics.uci.edu/ml/datasets/Student+Performance"
  
theFormula <- function(theSelected) {
  f <- returnFeaturesSelected(theSelected)
  return( f )
}

loadCSV <- function(URL) {
  csv <- tryCatch( {read.csv(text=getURL(URL,ssl.verifypeer=0L, followlocation=1L),sep=';')},
                          error=function(cond) {return(read.csv(URL,sep=';'))})    
  return(csv)
}



SelectStudentData <- function(theData){
  if (theData == "Portugues") {
    studentData     <- loadCSV("https://raw.githubusercontent.com/ervachon/Developing_Data_Products/gh-pages/student/student-por.csv")   
  } else {
    studentData     <- loadCSV("https://raw.githubusercontent.com/ervachon/Developing_Data_Products/gh-pages/student/student-mat.csv")   
  }
  
  studentData$ageLevel <- as.factor(studentData$age)
  return(studentData)
  
}

myPredict <- function(myData,theSelectedFeatures){
  set.seed(12345)
  inTrain           <- createDataPartition(y=myData$G3, p=.8, list = FALSE)
  ourTrain          <- myData[inTrain,]
  ourTest           <- myData[-inTrain,]
  modelFit          <- train(as.formula(theFormula(theSelectedFeatures)), method = "glm",data=ourTrain)
  testPred          <- predict(modelFit,ourTest)
  results           <- cbind(ourTest$G3,testPred)
  colnames(results) <- c("Reference", "Prediction")
  return(as.data.frame(results))
}

    set.seed(12345)
        
    myData <- SelectStudentData(data)
    
     ggplot(myData(), aes(x=ageLevel,fill=sex))
            + geom_bar(binwidth=.5,position="dodge")
            + facet_grid(school ~ . )
            + xlab ("Age of the students")
            + scale_fill_manual(values=c( theColors(input$col1), theColors(input$col2)))
    
    myPred      <- myPredict(myData(), if (input$updSelFeat==0)
                                          {c(1:32)}
                                       else
                                          {isolate ({c(input$featuresSel1,input$featuresSel2)})}
                                      )
                            
    pred <- renderPlot( ggplot(myPred(), aes(x = Reference, y = Prediction)) 
                                   + geom_point(color='blue')
                                   + geom_abline(intercept=0,slope=1,colour='red') 
                                   + geom_smooth(color = 'green')
                             )
    
    