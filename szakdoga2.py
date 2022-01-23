import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from numpy import array
from sklearn.model_selection import KFold

# data sample

input_file = "aptos2019-blindness-detection/train.csv" # eredeti, teljes csv

input_file_0 = "aptos2019-blindness-detection/ertek_0.csv" # csak a 0-ás osztály csv

input_file_1 = "aptos2019-blindness-detection/ertek_1.csv" # csak az 1-es osztály csv

input_file_2 = "aptos2019-blindness-detection/ertek_2.csv" #csak a 2-es osztály csv

input_file_3 = "aptos2019-blindness-detection/ertek_3.csv" #csak a 3-as osztály csv

input_file_4 = "aptos2019-blindness-detection/ertek_4.csv" #csak a 4-es osztály csv

# összes adathalmaz beolvasva id/diagnosis oszlopokkal

df = pd.read_csv(input_file, header = 0) # eredeti, teljes csv

df_0 = pd.read_csv(input_file_0, header = 0) #csak a 0-ás osztály csv

df_1 = pd.read_csv(input_file_1, header = 0) # csak az 1-es osztály csv

df_2 = pd.read_csv(input_file_2, header = 0) #csak a 2-es osztály csv

df_3 = pd.read_csv(input_file_3, header = 0) #csak a 3-as osztály csv

df_4 = pd.read_csv(input_file_4, header = 0) #csak a 4-es osztály csv

# eredeti, teljes csv

df_x = df.iloc[:, :-1].values # ID tömb
df_y = df.iloc[:, 1].values # diagnosis tömb


# FELOZTAS 60-20-20 train-test-val arányban

train_szazalek = 0.60
validation_szazalek = 0.20
test_szazalek = 0.20

# 0 felosztasa 3 reszre

df_x_0 = df_0.iloc[:, :-1].values # ID tömb
df_y_0 = df_0.iloc[:, 1].values # osztály tömb



x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(df_x_0, df_y_0, test_size= 1 - train_szazalek)

x_val_0, x_test_0, y_val_0, y_test_0 = train_test_split(x_test_0, y_test_0, test_size= test_szazalek/(test_szazalek + validation_szazalek))

# 1 felosztasa 3 reszre

df_x_1 = df_1.iloc[:, :-1].values # ID tömb
df_y_1 = df_1.iloc[:, 1].values # osztály tömb

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df_x_1, df_y_1, test_size= 1 - train_szazalek)

x_val_1, x_test_1, y_val_1, y_test_1 = train_test_split(x_test_1, y_test_1, test_size= test_szazalek/(test_szazalek + validation_szazalek))

# 2 felosztasa 3 reszre

df_x_2 = df_2.iloc[:, :-1].values # ID tömb
df_y_2 = df_2.iloc[:, 1].values # osztály tömb

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(df_x_2, df_y_2, test_size= 1 - train_szazalek)

x_val_2, x_test_2, y_val_2, y_test_2 = train_test_split(x_test_2, y_test_2, test_size= test_szazalek/(test_szazalek + validation_szazalek))

# 3 felosztasa 3 reszre

df_x_3 = df_3.iloc[:, :-1].values # ID tömb
df_y_3 = df_3.iloc[:, 1].values # osztály tömb

x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(df_x_3, df_y_3, test_size= 1 - train_szazalek)

x_val_3, x_test_3, y_val_3, y_test_3 = train_test_split(x_test_3, y_test_3, test_size= test_szazalek/(test_szazalek + validation_szazalek))

# 4 felosztasa 3 reszre

df_x_4 = df_4.iloc[:, :-1].values # ID tömb
df_y_4 = df_4.iloc[:, 1].values # osztály tömb

x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(df_x_4, df_y_4, test_size= 1 - train_szazalek)

x_val_4, x_test_4, y_val_4, y_test_4 = train_test_split(x_test_4, y_test_4, test_size= test_szazalek/(test_szazalek + validation_szazalek))


# 0 train-test-val dataframes


df_uj_0_train = pd.DataFrame(data = np.column_stack((x_train_0, y_train_0)), columns=['id', 'diagnosis'])
df_uj_0_test = pd.DataFrame(data = np.column_stack((x_test_0, y_test_0)), columns=['id', 'diagnosis'])
df_uj_0_val = pd.DataFrame(data = np.column_stack((x_val_0, y_val_0)), columns=['id', 'diagnosis'])

# 1 train-test-val dataframes


df_uj_1_train = pd.DataFrame(data = np.column_stack((x_train_1, y_train_1)), columns=['id', 'diagnosis'])
df_uj_1_test = pd.DataFrame(data = np.column_stack((x_test_1, y_test_1)), columns=['id', 'diagnosis'])
df_uj_1_val = pd.DataFrame(data = np.column_stack((x_val_1, y_val_1)), columns=['id', 'diagnosis'])

# 2 train-test-val dataframes


df_uj_2_train = pd.DataFrame(data = np.column_stack((x_train_2, y_train_2)), columns=['id', 'diagnosis'])
df_uj_2_test = pd.DataFrame(data = np.column_stack((x_test_2, y_test_2)), columns=['id', 'diagnosis'])
df_uj_2_val = pd.DataFrame(data = np.column_stack((x_val_2, y_val_2)), columns=['id', 'diagnosis'])

# 3 train-test-val dataframes


df_uj_3_train = pd.DataFrame(data = np.column_stack((x_train_3, y_train_3)), columns=['id', 'diagnosis'])
df_uj_3_test = pd.DataFrame(data = np.column_stack((x_test_3, y_test_3)), columns=['id', 'diagnosis'])
df_uj_3_val = pd.DataFrame(data = np.column_stack((x_val_3, y_val_3)), columns=['id', 'diagnosis'])

# 4 train-test-val dataframes


df_uj_4_train = pd.DataFrame(data = np.column_stack((x_train_4, y_train_4)), columns=['id', 'diagnosis'])
df_uj_4_test = pd.DataFrame(data = np.column_stack((x_test_4, y_test_4)), columns=['id', 'diagnosis'])
df_uj_4_val = pd.DataFrame(data = np.column_stack((x_val_4, y_val_4)), columns=['id', 'diagnosis'])

# concatenated train-test-val

concatenated_train = pd.concat([df_uj_0_train, df_uj_1_train, df_uj_2_train, df_uj_3_train, df_uj_4_train])

concatenated_test = pd.concat([df_uj_0_test, df_uj_1_test, df_uj_2_test, df_uj_3_test, df_uj_4_test])

concatenated_val = pd.concat([df_uj_0_val, df_uj_1_val, df_uj_2_val, df_uj_3_val, df_uj_4_val])

# osztaly darab ellenorzes train

concatenated_train_id = concatenated_train.iloc[:, :-1].values # ID tömb
concatenated_train_diagnosis = concatenated_train.iloc[:, 1].values # osztály tömb

tr_darab_0 = 0
tr_darab_1 = 0
tr_darab_2 = 0
tr_darab_3 = 0
tr_darab_4 = 0


for i in concatenated_train_diagnosis:
    if i == 0:
        tr_darab_0 += 1
    if i == 1:
        tr_darab_1 += 1
    if i == 2:
        tr_darab_2 += 1
    if i == 3:
        tr_darab_3 += 1
    if i == 4:
        tr_darab_4 +=1

# osztaly darab ellenorzes test

concatenated_test_id = concatenated_test.iloc[:, :-1].values # ID tömb
concatenated_test_diagnosis = concatenated_test.iloc[:, 1].values # osztály tömb

te_darab_0 = 0
te_darab_1 = 0
te_darab_2 = 0
te_darab_3 = 0
te_darab_4 = 0


for i in concatenated_test_diagnosis:
    if i == 0:
        te_darab_0 += 1
    if i == 1:
        te_darab_1 += 1
    if i == 2:
        te_darab_2 += 1
    if i == 3:
        te_darab_3 += 1
    if i == 4:
        te_darab_4 +=1
        
# osztaly darab ellenorzes val

concatenated_val_id = concatenated_val.iloc[:, :-1].values # ID tömb
concatenated_val_diagnosis = concatenated_val.iloc[:, 1].values # osztály tömb

v_darab_0 = 0
v_darab_1 = 0
v_darab_2 = 0
v_darab_3 = 0
v_darab_4 = 0


for i in concatenated_val_diagnosis:
    if i == 0:
        v_darab_0 += 1
    if i == 1:
        v_darab_1 += 1
    if i == 2:
        v_darab_2 += 1
    if i == 3:
        v_darab_3 += 1
    if i == 4:
        v_darab_4 +=1






##GYAKORLAS##

#train_ratio = 0.6
#validation_ratio = 0.2
#test_ratio = 0.2

#train is now 60% of the entire data set

#x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=1 - train_ratio)

# test is now 20% of the initial data set
# validation is now 20% of the initial data set

#x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

#db_0 = 0
#db_1 = 0
#db_2 = 0
#db_3 = 0
#db_4 = 0


#for i in y_test:
    #if i == 0:
        #db_0 += 1
    #if i == 1:
        #db_1 += 1
    #if i == 2:
        #db_2 += 1
    #if i == 3:
        #db_3 += 1
    #else:
        #db_4 +=1

#print(round(db_0 / len(y_train), 2), round(db_1 / len(y_train), 2), round(db_2 / len(y_train), 2), round(db_3 / len(y_train), 2), round(db_4 / len(y_train), 2))

#plt.hist(y_train)
#plt.xlim(0, 4)
#plt.show()

#plt.hist(y_test)
#plt.xlim(0, 4)
#plt.show()

#plt.hist(y_val)
#plt.xlim(0, 4)
#plt.show()

# prepare cross validation

#kfold = KFold(3, True, 1)

# enumerate splits

#for train, test in kfold.split(df):
	#print('train: %s, test: %s' % (df[train], df[test]))

#for train, test, val in kfold.split(df):
	#print('train: %s, test: %s', 'val: %s' % (df[train], df[test], df[val]))
    
    
    
#kf = KFold(n_splits=3)

#for train_index, test_index in kf.split(df):
    #print("TRAIN:", train_index, "VALIDATION:", test_index)
    #X_train, X_test = df_x[train_index], df_x[test_index]
    #y_train, y_test = df_y[train_index], df_y[test_index]

