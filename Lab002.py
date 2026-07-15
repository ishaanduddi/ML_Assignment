##A1

import pandas as pd
import numpy as np

# Load the required columns from the Excel sheet
df = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab02.xlsx",sheet_name="Purchase data",usecols="A:E")
def A1(df):
    # Feature matrix X
    X = df.iloc[:, 1:4].values

    # Output vector y
    y = df.iloc[:, 4:5].values

    # Calculate the rank of X
    rank = np.linalg.matrix_rank(X)


    psuedoinverse=np.linalg.pinv(X)
    cost=psuedoinverse @ y
    return X,y,X.shape,y.shape,rank,psuedoinverse,cost
X,y,X.shape,y.shape,rank,psuedoinverse,cost=A1(df)
print("Feature Matrix (X):")
print(X)
print("\nOutput Vector (y):")
print(y)
print("\nShape of X:", X.shape)
print("\nShape of y:", y.shape)
print("\nAll Pseudo Inverse",psuedoinverse)
print("\nRank of Feature Matrix X:", rank)
print("\nCost of Candy", cost[0,0])
print("\nCost of Mangoes", cost[1,0])
print("\nCost of Milk", cost[2,0])

##A2
def A2(df):
    df["classifier"] = df["Payment (Rs)"].apply(lambda x: "RICH" if x > 200 else "POOR")
    return df

classifier=A2(df)
print(classifier)

##A3
df1= pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab02.xlsx",sheet_name="IRCTC Stock Price",usecols="D")
def A3(df):
    return df

ans=A3(df1)
print(ans)
