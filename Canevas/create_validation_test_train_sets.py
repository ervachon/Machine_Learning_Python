# -*- coding: utf-8 -*-
"""
Created on 2016/02/03

@author: eriva

execfile('create_validation_test_train_sets.py')

"""
execfile('libraries.py')

print("> load raw data")
completePath = os.path.join(os.getcwd(), data_raw_dir,data_raw)
myDF = pd.read_csv(completePath,sep=separator,low_memory=False, na_values=na_value)
myDF = myDF.drop_duplicates()
myDF = myDF.reindex(np.random.permutation(myDF.index)).reset_index(drop=True)

# transform Y into int
myDF["Contraceptive_method_used"] = myDF["Contraceptive_method_used"].replace(['No_use', 'Short_term', 'Long_term'], [0, 1, 2]) 

# add a key column in first poosition
myDF['key'] = myDF.index
colnames = myDF.columns.tolist()
colnames = colnames[-1:] + colnames[:-1]
myDF = myDF[colnames]

print("> split validation / test train")
validation_length = int(len(myDF)*validation_ratio)
validation = myDF.iloc[0:validation_length,]
train_test = myDF.iloc[validation_length:,]

print("> save test train")
train_test.to_pickle(os.path.join(os.getcwd(), pickle_dir,train_test_pkl))
train_test.to_csv(os.path.join(os.getcwd(), data_dir,train_test_data),index = False)

print("> save validation")
validation.iloc[:,-1].to_pickle(os.path.join(os.getcwd(), pickle_dir,validation_pkl_Y))
validation.iloc[:,-1].to_csv(os.path.join(os.getcwd(), data_dir,validation_data_Y),index = False)

validation.iloc[:,:-1].to_pickle(os.path.join(os.getcwd(), pickle_dir,validation_pkl_X))
validation.iloc[:,:-1].to_csv(os.path.join(os.getcwd(), data_dir,validation_data_X),index = False)
