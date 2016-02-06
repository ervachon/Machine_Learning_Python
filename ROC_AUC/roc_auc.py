# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 14:14:57 2016

@author: eriva

os.chdir("D:\\_GIT_\\Machine_Learning_Python\\ROC_AUC")

+ -> 1
- -> 0

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

myDF = pd.read_csv(os.path.join(os.getcwd(), "data_test.csv"),sep=";",low_memory=False)
myDF["y_true"] = myDF["y_true"].astype('float64')
myDF["y_proba"] = myDF["y_proba"].astype('float64')

NbN = len(myDF[(myDF.y_true == 0)])
NbP = len(myDF[(myDF.y_true == 1)])

plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.plot([1,0,0,1,],[0,0,1,1,] , 'g-')   

lP=[[1.],[0.],[0.]]

for seuil in np.arange(0,1.05,0.05):
   sub = myDF[(myDF.y_proba >= seuil)]
   TVP = 1.*len(sub[(sub.y_true == 1)])/NbP
   FVP = 1.*len(sub[(sub.y_true == 0)])/NbN
   print("> seuil : " + str(seuil) + ", (" + str(FVP) + "," + str(TVP) + ")")
   
   x1 = FVP
   x2 = lP[0][len(lP[0])-1]
   y1 = TVP
   y2 = lP[1][len(lP[0])-1]
   aire = np.abs(1.*np.min([y1,y2])*(x1-x2)) + np.abs((y1-y2)*(x1-x2)/2.)
   lP[0] = lP[0] + [FVP]
   lP[1] = lP[1] + [TVP]
   lP[2] = lP[2] + aire

lP[0] = lP[0] + [0.]
lP[1] = lP[1] + [0.]

print( "> AUC = " + str(np.sum(lP[2])))

plt.plot(lP[0], lP[1], linestyle='-')   
plt.plot([0,1],[0,1] , 'r-')   
plt.show()