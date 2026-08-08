import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
import math 
from scipy.spatial.distance import *

dfall = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab03.xlsx", sheet_name="marketing_campaign")

df = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab03.xlsx", sheet_name="marketing_campaign",usecols="C:F")
dfplot = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab03.xlsx", sheet_name="marketing_campaign",usecols="I:J")
df1 = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab03.xlsx", sheet_name="marketing_campaign",usecols="I")

##A1
def A1():
    return "ID - Nominal\nYear_Birth - Interval\nEducation - Ordinal\nMarital_Status - Nominal\nIncome - Ratio\nKidhome - Ratio\nTeenhome - Ratio\nDt_Customer - Interval\nRecency - Ratio\nMntWines - Ratio\nMntFruits - Ratio\nMntMeatProducts - Ratio\nMntFishProducts - Ratio\nMntSweetProducts - Ratio\nMntGoldProds - Ratio\nNumDealsPurchases - Ratio\nNumWebPurchases - Ratio\nNumCatalogPurchases - Ratio\nNumStorePurchases - Ratio\nNumWebVisitsMonth - Ratio\nAcceptedCmp1 - Nominal\nAcceptedCmp2 - Nominal\nAcceptedCmp3 - Nominal\nAcceptedCmp4 - Nominal\nAcceptedCmp5 - Nominal\nComplain - Nominal\nZ_CostContact - Ratio\nZ_Revenue - Ratio\nResponse - Nominal"

a1=A1()
print(a1)

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

##A4
a=[5,6,7,8]
b=[1,2,3,4]
def euclidean(a,b):
    euc=0
    for i in range(len(a)):
            if(a[i]>b[i]):
                euc+=(a[i]-b[i])**2
            else:
                euc+=(b[i]-a[i])**2
    euc=math.sqrt(euc)
    return euc

euc=euclidean(a,b)
print("Eucledian value will be :- ",euc)

def minkowskii(a,b,p):
    mink=0
    for i in range(len(a)):
            if(a[i]>b[i]):
                mink+=(a[i]-b[i])**p
            else:
                mink+=(b[i]-a[i])**p
    mink=math.pow(mink,1/p)
    return mink

p=int(input("Enter p value :- "))
mink=minkowskii(a,b,p)
print("minkowskii value will be :- ",mink)

def manhattan(a,b):
    man=0
    for i in range(len(a)):
            if(a[i]>b[i]):
                man+=(a[i]-b[i])
            else:
                man+=(b[i]-a[i])
    return man
man=manhattan(a,b)
print("Manhattan value will be :- ",man)

##A5
fv1=dfplot.iloc[0].astype(int).tolist()
fv2=dfplot.iloc[1].astype(int).tolist()
pval=[]
dist=[]

for i in range(1,11):
    d = minkowskii(fv1, fv2, i)
    pval.append(i)
    dist.append(d)
    print("p =", i, "Distance =", d)

plt.plot(pval,dist,marker='o')
plt.show()


##A6
distt = minkowski(a, b, p)

print("Scipy minkowskii Distance :-", distt)

if distt == mink:

    print("equal")
else:
     print("not equal")

##A7
def A7dp(a, b):
    product = 0
    for i in range(len(a)):
        product += a[i] * b[i]
    return product

product = A7dp(a, b)
print("Dot Product :-", product)

def A7en(a):
    norm = 0
    for i in range(len(a)):
        norm += a[i] ** 2
    norm = math.sqrt(norm)
    return norm

normA = A7en(a)
normB = A7en(b)

print("Length of A :-", normA)
print("Length of B :-", normB)

dot_numpy = np.dot(a, b)
normA_numpy = np.linalg.norm(a)
normB_numpy = np.linalg.norm(b)

print("NumPy Dot Product :-", dot_numpy)
print("NumPy Length of A :-", normA_numpy)
print("NumPy Length of B :-", normB_numpy)

if product == dot_numpy:
    print("equal")
else:
     print("not equal")


if normA == normA_numpy:
    print("equal")
else:
     print("not equal")


if normB == normB_numpy:
    print("equal")
else:
     print("not equal")


##A8
def mean(numbers):
    length = len(numbers)
    sum = 0
    for i in range(len(numbers)):
        sum += numbers[i]
    avg = sum / length
    return avg

def variance(numbers):
    avg = mean(numbers)
    length = len(numbers)
    var = 0
    for i in range(len(numbers)):
        var += (numbers[i] - avg) ** 2
    var = var / length
    return var

def standarddeviation(numbers):
    var = variance(numbers)
    sd = math.sqrt(var)
    return sd

def A8(df):
    for column in df.columns:
        values = df[column].tolist()

        avg = mean(values)
        var = variance(values)
        sd = standarddeviation(values)
        print("Mean :", avg)
        print("Variance :", var)
        print("Standard Deviation :", sd)
        print()

A8(dfplot)


##A9
df_numeric = dfall.select_dtypes(include=['number'])

mean_numpy = np.mean(df_numeric)
std_numpy = np.std(df_numeric)
meann = []
stdd = []

for column in df_numeric.columns:
    values = df_numeric[column].tolist()
    meann.append(mean(values))
    stdd.append(standarddeviation(values))

print("Mean using my function :-",meann)

print("Mean using NumPy :-",mean_numpy)

print("Standard Deviation using my function :-",stdd)

print("Standard Deviation using NumPy :-",std_numpy)

if meann == list(mean_numpy):
    print("equal")
else:
    print("not equal")

if stdd == list(std_numpy):
    print("equal")
else:
    print("not equal")

##A10
feature = df1.iloc[:, 0].tolist()

avg = mean(feature)
var = variance(feature)

print("Mean :", avg)
print("Variance :", var)

hist, bins = np.histogram(feature)

print("Histogram Frequencies :", hist)
print("Histogram Bins :", bins)

plt.hist(feature)
plt.show()