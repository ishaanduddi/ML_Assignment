import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab02.xlsx", sheet_name="Purchase data", usecols="A:E")
df1 = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab02.xlsx", sheet_name="IRCTC Stock Price")
df2 = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab02.xlsx", sheet_name="thyroid0387_UCI")


##A1
def A1(df):
    X = df.iloc[:, 1:4].values

    y = df.iloc[:, 4:5].values

    rank = np.linalg.matrix_rank(X)

    pseudoinverse = np.linalg.pinv(X)
    cost = pseudoinverse @ y
    return X, y, X.shape, y.shape, rank, pseudoinverse, cost

print("A1")
X, y, X_shape, y_shape, rank, pseudoinverse, cost = A1(df)
print("Feature Matrix (X):")
print(X)
print("\nOutput Vector (y):")
print(y)
print("\nShape of X:", X_shape)
print("Shape of y:", y_shape)
print("\nAll Pseudo Inverse:\n", pseudoinverse)
print("\nRank of Feature Matrix X:", rank)
print("\nCost of Candy:", cost[0, 0])
print("Cost of Mangoes:", cost[1, 0])
print("Cost of Milk:", cost[2, 0])


##A2
def A2(df):
    df["classifier"] = df["Payment (Rs)"].apply(lambda x: "RICH" if x > 200 else "POOR")

    X = df.iloc[:, 1:4].values
    y = df["classifier"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return df, accuracy

print("A2")
classified_df, accuracy = A2(df)
print(classified_df)
print("\nClassifier Accuracy (kNN, k=3):", accuracy)


##A3
def mean_custom(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)


def variance_custom(data):
    m = mean_custom(data)
    total = 0
    for value in data:
        total += (value - m) ** 2
    return total / len(data)


def A3(df1):
    price = df1["Price"].values

    # Mean and variance using numpy
    np_mean = np.mean(price)
    np_var = np.var(price)

    # Mean and variance using custom functions
    custom_mean = mean_custom(price)
    custom_var = variance_custom(price)

    np_times = []
    custom_times = []
    for _ in range(10):
        t0 = time.time()
        np.mean(price)
        np.var(price)
        np_times.append(time.time() - t0)

        t0 = time.time()
        mean_custom(price)
        variance_custom(price)
        custom_times.append(time.time() - t0)

    avg_np_time = mean_custom(np_times)
    avg_custom_time = mean_custom(custom_times)

    wed_price = df1[df1["Day"] == "Wed"]["Price"].values
    wed_mean = mean_custom(wed_price)

    apr_price = df1[df1["Month"] == "Apr"]["Price"].values
    apr_mean = mean_custom(apr_price)

    is_loss = list(map(lambda x: x < 0, df1["Chg%"]))
    prob_loss = sum(is_loss) / len(is_loss)

    wed_df = df1[df1["Day"] == "Wed"]
    is_profit_wed = list(map(lambda x: x > 0, wed_df["Chg%"]))
    prob_profit_wed = sum(is_profit_wed) / len(wed_df)

    cond_prob_profit_given_wed = prob_profit_wed

    return (np_mean, np_var, custom_mean, custom_var, avg_np_time, avg_custom_time,
            wed_mean, apr_mean, prob_loss, prob_profit_wed, cond_prob_profit_given_wed)


def A3_scatter(df1):
    plt.figure(figsize=(10, 6))
    plt.scatter(df1["Day"], df1["Chg%"])
    plt.xlabel("Day of the Week")
    plt.ylabel("Chg%")
    plt.title("Chg% vs Day of the Week")
    plt.savefig("A3_scatter.png")
    plt.close()

print("A3")
(np_mean, np_var, custom_mean, custom_var, avg_np_time, avg_custom_time,
 wed_mean, apr_mean, prob_loss, prob_profit_wed, cond_prob_profit_given_wed) = A3(df1)
print("Mean (numpy):", np_mean, " Mean:", custom_mean)
print("Variance (numpy):", np_var, " Variance:", custom_var)
print("Avg time (numpy):", avg_np_time, " Avg time :", avg_custom_time)
print("Population Mean:", np_mean, " Wednesday Mean:", wed_mean)
print("Population Mean:", np_mean, " April Mean:", apr_mean)
print("Probability of Loss:", prob_loss)
print("Probability of Profit on Wednesday:", prob_profit_wed)
print("P(Profit | Wednesday):", cond_prob_profit_given_wed)
A3_scatter(df1)


##A4
def A4(df2):
    datatypes = df2.dtypes

    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df2.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_counts = (df2 == "?").sum()

    df_clean = df2.replace("?", np.nan)
    for col in ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    numeric_range = {}
    mean_var = {}
    for col in ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]:
        numeric_range[col] = (df_clean[col].min(), df_clean[col].max())
        mean_var[col] = (df_clean[col].mean(), df_clean[col].var())

    return datatypes, numeric_cols, categorical_cols, missing_counts, numeric_range, mean_var, df_clean

print("A4")
datatypes, numeric_cols, categorical_cols, missing_counts, numeric_range, mean_var, df_clean = A4(df2)
print("Datatypes:\n", datatypes)
print("\nNumeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)
print("\nMissing Value Counts:\n", missing_counts[missing_counts > 0])
print("\nNumeric Ranges:", numeric_range)
print("\nMean and Variance:", mean_var)


##A5
def A5(df2):
    binary_cols = ["on thyroxine", "query on thyroxine", "on antithyroid medication", "sick",
                   "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid",
                   "query hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych"]

    vec1 = df2.iloc[0][binary_cols].apply(lambda x: 1 if x == "t" else 0).values
    vec2 = df2.iloc[1][binary_cols].apply(lambda x: 1 if x == "t" else 0).values

    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f00 = np.sum((vec1 == 0) & (vec2 == 0))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))

    jc = f11 / (f01 + f10 + f11)
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)

    return jc, smc

print("A5")
jc, smc = A5(df2)
print("Jaccard Coefficient:", jc)
print("Simple Matching Coefficient:", smc)


##A6
def A6(df_clean):
    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

    vec1 = df_clean.iloc[0][numeric_cols].fillna(0).values.astype(float)
    vec2 = df_clean.iloc[1][numeric_cols].fillna(0).values.astype(float)

    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

print("A6")
cos_sim = A6(df_clean)
print("Cosine Similarity:", cos_sim)


##A7
def A7(df2, df_clean):
    binary_cols = ["on thyroxine", "query on thyroxine", "on antithyroid medication", "sick",
                   "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid",
                   "query hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych"]
    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

    n = 20
    bin_data = df2.iloc[:n][binary_cols].apply(lambda col: col.map(lambda x: 1 if x == "t" else 0)).values
    num_data = df_clean.iloc[:n][numeric_cols].fillna(0).values.astype(float)

    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))
    cos_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            v1 = bin_data[i]
            v2 = bin_data[j]
            f11 = np.sum((v1 == 1) & (v2 == 1))
            f00 = np.sum((v1 == 0) & (v2 == 0))
            f10 = np.sum((v1 == 1) & (v2 == 0))
            f01 = np.sum((v1 == 0) & (v2 == 1))

            jc_matrix[i, j] = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0
            smc_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11)

            nv1 = num_data[i]
            nv2 = num_data[j]
            denom = np.linalg.norm(nv1) * np.linalg.norm(nv2)
            cos_matrix[i, j] = np.dot(nv1, nv2) / denom if denom != 0 else 0

    return jc_matrix, smc_matrix, cos_matrix


def A7_heatmap(jc_matrix, smc_matrix, cos_matrix):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sns.heatmap(jc_matrix, annot=False, ax=axes[0])
    axes[0].set_title("Jaccard Coefficient")
    sns.heatmap(smc_matrix, annot=False, ax=axes[1])
    axes[1].set_title("Simple Matching Coefficient")
    sns.heatmap(cos_matrix, annot=False, ax=axes[2])
    axes[2].set_title("Cosine Similarity")
    plt.savefig("A7_heatmap.png")
    plt.close()


print("A7")
jc_matrix, smc_matrix, cos_matrix = A7(df2, df_clean)
A7_heatmap(jc_matrix, smc_matrix, cos_matrix)
print("Heatmaps saved to A7_heatmap.png")


##A8
def A8(df_clean):
    df_imputed = df_clean.copy()

    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    categorical_cols = [c for c in df_imputed.columns if c not in numeric_cols and c != "Record ID"]

    for col in numeric_cols:
        q1 = df_imputed[col].quantile(0.25)
        q3 = df_imputed[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        has_outliers = ((df_imputed[col] < lower) | (df_imputed[col] > upper)).any()

        if has_outliers:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        else:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

    for col in categorical_cols:
        mode_val = df_imputed[col].mode()[0]
        df_imputed[col] = df_imputed[col].fillna(mode_val)

    return df_imputed

print("A8")
df_imputed = A8(df_clean)
print("Remaining missing values after imputation:", df_imputed.isnull().sum().sum())


##A9
def A9(df_imputed):
    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

    scaler = MinMaxScaler()
    df_normalized = df_imputed.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df_imputed[numeric_cols])

    return df_normalized

print("A9")
df_normalized = A9(df_imputed)
print(df_normalized.head())