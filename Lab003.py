import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import *
from sklearn.model_selection import *
from sklearn.preprocessing import *

df = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab03.xlsx", sheet_name="marketing_campaign",usecols="C:F")
    
##A1
print(" ")

##A2
def A2_label(df):
    column= df["Education"].unique()
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Education'])
    return column,df
col,label=A2_label(df)
print(col)
print(label)

def A2_OneHot(df):
    column=df["Marital_Status"]
    one_hot = pd.get_dummies(column, dtype=int)
    return one_hot
onehot=A2_OneHot(df)
print(onehot)

##A3
a=[5,6,7,8]
b=[1,2,3,4]
def minkowski(a,b):
    for i in range(a):
            if(a[i]>b[i]):
                    a[i]-b[i]
            else:
                 b[i]-a[i]

