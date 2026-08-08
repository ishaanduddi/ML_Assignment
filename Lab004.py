import pandas as pd

# -------------------- Read Excel File --------------------

# Replace the filename with your Excel file name
file_path = r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab04.xlsx"

# Read Excel
df = pd.read_excel(file_path,sheet_name="marketing_campaign")

# Display all column names
print("Columns in Excel:")
print(df.columns)

# ---------------------------------------------------------
# Select a categorical column
# Change "Education" to any categorical column if needed
# Example: "Marital_Status", "Education"
# ---------------------------------------------------------
print("A2")
column_name = "Marital_Status"

values = df[column_name].dropna().tolist()

print("\nOriginal Data")
print(values)


# =========================================================
# LABEL ENCODING
# =========================================================

def label_encoding(data):

    label_dict = {}
    encoded_list = []

    label = 0

    # Assign labels only for unique values
    for item in data:

        if item not in label_dict:
            label_dict[item] = label
            label += 1

    # Encode the original data
    for item in data:
        encoded_list.append(label_dict[item])

    print("\n===============================")
    print("LABEL ENCODING")
    print("===============================")

    print("\nLabel Dictionary")
    for key, value in label_dict.items():
        print(key, ":", value)

    print("\nEncoded Values")

    for i in range(len(data)):
        print(data[i], "->", encoded_list[i])

    return label_dict, encoded_list


# =========================================================
# ONE HOT ENCODING
# =========================================================

def one_hot_encoding(data):

    unique_values = []

    # Find all distinct values
    for item in data:
        if item not in unique_values:
            unique_values.append(item)

    encoded_rows = []

    # Create one-hot vectors
    for item in data:

        row = []

        for value in unique_values:

            if item == value:
                row.append(1)
            else:
                row.append(0)

        encoded_rows.append(row)

    # Convert into DataFrame
    one_hot_df = pd.DataFrame(encoded_rows, columns=unique_values)

    print("\n===============================")
    print("ONE HOT ENCODING")
    print("===============================\n")

    print(one_hot_df)

    return one_hot_df


# =========================================================
# Function Calls
# =========================================================

label_dict, encoded = label_encoding(values)

one_hot_table = one_hot_encoding(values)

print("A4")
import math

def minkowski(a, b, p):

    mink = 0

    for i in range(len(a)):

        if a[i] > b[i]:
            mink += (a[i] - b[i]) ** p
        else:
            mink += (b[i] - a[i]) ** p

    mink = math.pow(mink, 1/p)

    if p == 1:
        print("Manhattan Distance =", mink)
    elif p == 2:
        print("Euclidean Distance =", mink)
    else:
        print("Minkowski Distance (p =", p, ") =", mink)

    return mink


# Example
a = [2, 4, 6]
b = [5, 8, 9]

# Manhattan Distance
minkowski(a, b, 1)

# Euclidean Distance
minkowski(a, b, 2)

# Generalized Minkowski Distance
minkowski(a, b, 3)

