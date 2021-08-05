# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:36:59 2021

@author: elite
"""

import pandas as pd

file1 = pd.read_excel('./file1.xlsx') ## THIS IS OLD DATA
file2 = pd.read_excel('./file2.xlsx') ## THIS IS NEW DATA

df = pd.concat([file1,file2])
df = df.fillna(0)
#print(df)

df = df.reset_index(drop=True)
#print(df)

df_grpby = df.groupby(list(df.columns))
idx = [x[0] for x in df_grpby.groups.values() if len(x) == 1 and x[0] >= len(file1)]
idx2 = [x[0] for x in df_grpby.groups.values() if len(x) == 1 and x[0] < len(file1)]

#print(idx)


#print("THIS IS THE FORMAT:")
print("----------OLD DATA VALUES----------")
print(df.reindex(idx2))
print("----------NEW DATA VALUES----------")
print(df.reindex(idx))


# print([x[0] for x in df_grpby.groups.values()])
