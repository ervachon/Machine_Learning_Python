# -*- coding: utf-8 -*-
"""
Created on 2016/02/03

@author: eriva

execfile('create_kfold.py')

"""
execfile('libraries.py')

print("> load test train")
completePath = os.path.join(os.getcwd(), data_dir,train_test_data)
myDF = pd.read_csv(completePath,sep=separator,low_memory=False, na_values=na_value)
myDF = myDF.drop_duplicates()
myDF = myDF.reindex(np.random.permutation(myDF.index)).reset_index(drop=True)
labels = myDF.iloc[:,-1]

list_kfolds = np.zeros(len(myDF), dtype=int)
list_kfolds.fill(-1) 

stratif = StratifiedKFold(labels,n_folds = kfold, random_state=1994)

for i, (tri,cvi) in enumerate(stratif): list_kfolds[cvi]=i

myDFKfold=myDF[['key']]
myDFKfold['kfold'] = list_kfolds
myDFKfold = myDFKfold.sort_values(by='key').reset_index(drop=True)

myDFKfold.to_pickle(os.path.join(os.getcwd(), pickle_dir,kfolds_pkl))
myDFKfold.to_csv(os.path.join(os.getcwd(), data_dir,kfolds_data),index = False)
