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

seuil = 0.01

def returnAUCTab(unDF):
    NbN = len(unDF[(unDF.y_true == 0)])
    NbP = len(unDF[(unDF.y_true == 1)])
    lP=pd.DataFrame({"x":[1.], "y":[0.], "aire":[0.], "seuil" :[0.]})
    
    for elt in np.arange(0.+seuil ,1.+seuil ,seuil ):
       sub = unDF[(unDF.y_proba >= elt)]
       TVP = 1.*len(sub[(sub.y_true == 1)])/NbP
       FVP = 1.*len(sub[(sub.y_true == 0)])/NbN
       #print("> seuil : " + str(elt) + ", (" + str(FVP) + "," + str(TVP) + ")")
       x1 = FVP
       x2 = lP.x[len(lP)-1]
       y1 = TVP
       y2 = lP.y[len(lP)-1]
       aire = np.abs(1.*np.min([y1,y2])*(x1-x2)) + np.abs((y1-y2)*(x1-x2)/2.)
       lP.loc[len(lP)]={"x":FVP, "y":TVP, "aire":aire, "seuil":elt}

    print('OK')
    return(lP)

myDF1 = pd.read_csv(os.path.join(os.getcwd(), "data_test_1.csv"),sep=";",low_memory=False)
myDF1["y_true"] = myDF1["y_true"].astype('float64')
myDF1["y_proba"] = myDF1["y_proba"].astype('float64')

myDF2 = pd.read_csv(os.path.join(os.getcwd(), "data_test_2.csv"),sep=";",low_memory=False)
myDF2["y_true"] = myDF2["y_true"].astype('float64')
myDF2["y_proba"] = myDF2["y_proba"].astype('float64')

#les axes
plt.axis([-0.05, 1.05, -0.05, 1.05])

#
plt.plot([1,0,0,1,],[0,0,1,1,] , 'k-')   

lP1=returnAUCTab(myDF1)
lP2=returnAUCTab(myDF2)

plt.plot(lP1.x.values, lP1.y.values, linestyle='-')   
plt.plot(lP2.x.values, lP2.y.values, linestyle='-')   

plt.plot([0,1],[0,1] , 'r-')   
plt.show()

print( "> AUC 1 (ref)      = " + str(np.sum(lP1.aire.values)))
print( "> AUC 2 (ameliore) = " + str(np.sum(lP2.aire.values)))

# merge les deux
lP1.columns = ['aire_1','seuil','x_1','y_1']
lP2.columns = ['aire_2','seuil','x_2','y_2']
lP = lP1.merge(lP2,on='seuil')

pront(lP[(lP.aire_1>lP.aire_2)])