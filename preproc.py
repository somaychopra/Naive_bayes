import pandas as pd
import numpy as np
# Location : 3
# Gender : 4
# Age : 5
fl = pd.read_csv("Train_D.csv",delimiter=',')
# fl["gender"] = pd.isnull(fl["gender"])
# print(fl)

for column in ['location', 'country', 'gender', 'visiting Wuhan', 'from Wuhan']:
    fl[column].fillna(fl[column].mode()[0], inplace=True)

for column in ['age']:
    fl[column].fillna(int(fl[column].mean()), inplace=True)

fl["country"] = fl["country"].astype('category')# 38 different types of countries
fl["country"] = fl["country"].cat.codes

fl["gender"] = fl["gender"].astype('category') # 2 types of gender
fl["gender"] = fl["gender"].cat.codes

fl["location"] = fl["location"].astype('category') # 156 different types of locations
fl["location"] = fl["location"].cat.codes

fl["death"] = fl["death"].replace(to_replace='[0-9][0-9]*/[0-9][0-9]/[0-9][0-9][0-9][0-9]|[0-9][0-9]*-[0-9][0-9]-[0-9][0-9][0-9][0-9]', value = '1', regex=True)
fl["death"] = fl["death"].astype('category') # 156 different types of locations
fl["death"] = fl["death"].cat.codes
print(fl["death"][992:1040])

fd = pd.DataFrame()
for column in ['location', 'country', 'gender', 'age', 'visiting Wuhan', 'from Wuhan', 'death'] :
	# print("ok")
	fd[column] = fl[column]

print(fd)
fd.to_csv('after_prepoc.csv',index=False)


# print(fl["death"])
# fl = fl.to_numpy()
# print(fl[-9:,4])
# print(fl[22,5])
# print(fl[169:172,-3])
# print(fl[200:250,-1])
# print(fl[50:60,-1])
# print(np.sum(fl[:,-1]))
# print(fl)