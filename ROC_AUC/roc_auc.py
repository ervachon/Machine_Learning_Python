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


lX=[1.]
lY=[0.]

for seuil in np.arange(0,1.05,0.05):
   sub = myDF[(myDF.y_proba >= seuil)]
   TVP = 1.*len(sub[(sub.y_true == 1)])/NbP
   FVP = 1.*len(sub[(sub.y_true == 0)])/NbN
   print("> seuil : " + str(seuil) + ", (" + str(FVP) + "," + str(TVP) + ")")
   lX = lX + [FVP]
   lY = lY + [TVP]

lX = lX + [0.]
lY = lY + [0.]
plt.plot(lX, lY, linestyle='-', marker='o')   
plt.plot([0,1],[0,1] , 'r-')   
plt.show()