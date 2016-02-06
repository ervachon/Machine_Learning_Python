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

myDF1 = pd.read_csv(os.path.join(os.getcwd(), "data_test_1.csv"),sep=";",low_memory=False)
myDF1["y_true"] = myDF1["y_true"].astype('float64')
myDF1["y_proba"] = myDF1["y_proba"].astype('float64')

NbN = len(myDF1[(myDF1.y_true == 0)])
NbP = len(myDF1[(myDF1.y_true == 1)])

plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.plot([1,0,0,1,],[0,0,1,1,] , 'g-')   

lP=pd.DataFrame({"x":[1.], "y":[0.], "aire":[0.]})

for seuil in np.arange(0,1.05,0.05):
   sub = myDF1[(myDF1.y_proba >= seuil)]
   TVP = 1.*len(sub[(sub.y_true == 1)])/NbP
   FVP = 1.*len(sub[(sub.y_true == 0)])/NbN
   print("> seuil : " + str(seuil) + ", (" + str(FVP) + "," + str(TVP) + ")")
   
   x1 = FVP
   x2 = lP.x[len(lP)-1]
   y1 = TVP
   y2 = lP.y[len(lP)-1]
   aire = np.abs(1.*np.min([y1,y2])*(x1-x2)) + np.abs((y1-y2)*(x1-x2)/2.)
  
   lP.loc[len(lP)]={"x":FVP, "y":TVP, "aire":aire}


print( "> AUC = " + str(np.sum(lP.aire.values)))

plt.plot(lP.x.values, lP.y.values, linestyle='-')   
plt.plot([0,1],[0,1] , 'r-')   
plt.show()