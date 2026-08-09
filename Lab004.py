# ============================================================
# MACHINE LEARNING LAB - A2 TO A11
# Marketing Campaign Dataset
# ============================================================

import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski as scipy_minkowski


# ============================================================
# LOAD DATASET ONCE
# ============================================================

file_path = r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab04.xlsx"

df = pd.read_excel(file_path, sheet_name="marketing_campaign")


# ============================================================
# COMMON FEATURES USED FOR DISTANCE CALCULATIONS
# ============================================================

features = [
    "Year_Birth",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]


# ============================================================
# A2
# Label Encoding and One-Hot Encoding
# ============================================================

print("\n\n")
print("############################################################")
print("A2 - LABEL ENCODING AND ONE-HOT ENCODING")
print("############################################################")


# ------------------------------------------------------------
# LABEL ENCODING FUNCTION
# ------------------------------------------------------------

def label_encoding(data, display=True):

    label_dict = {}
    encoded_list = []

    label = 0

    # Assign labels only to unique values
    for item in data:

        if item not in label_dict:
            label_dict[item] = label
            label += 1

    # Encode original data
    for item in data:
        encoded_list.append(label_dict[item])

    if display:

        print("\n===============================")
        print("LABEL ENCODING")
        print("===============================")

        print("\nLabel Encoding Dictionary")

        for key, value in label_dict.items():
            print(key, ":", value)

        print("\nEncoded Values")

        for i in range(len(data)):
            print(data[i], "->", encoded_list[i])

    return label_dict, encoded_list


# ------------------------------------------------------------
# ONE-HOT ENCODING FUNCTION
# ------------------------------------------------------------

def one_hot_encoding(data, display=True):

    unique_values = []

    # Find distinct values
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

    one_hot_df = pd.DataFrame(
        encoded_rows,
        columns=unique_values
    )

    if display:

        print("\n===============================")
        print("ONE HOT ENCODING")
        print("===============================")

        print(one_hot_df)

    return one_hot_df


# ------------------------------------------------------------
# A2 DATA
# ------------------------------------------------------------

column_name = "Marital_Status"

values = df[column_name].dropna().tolist()

print("\nOriginal Data:")
print(values)


label_dict, encoded = label_encoding(values)

one_hot_table = one_hot_encoding(values)


# ------------------------------------------------------------
# A2 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A2 FUNCTIONAL TESTING ---")

test_data = ["Apple", "Banana", "Apple", "Mango", "Banana"]

test_dictionary, test_encoded = label_encoding(
    test_data,
    display=False
)

test_one_hot = one_hot_encoding(
    test_data,
    display=False
)

print("Input:", test_data)
print("Label Dictionary:", test_dictionary)
print("Label Encoded:", test_encoded)
print("One-Hot Encoded:")
print(test_one_hot)


# ------------------------------------------------------------
# A2 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A2 UNIT TESTS CONDUCTED ---")

assert test_dictionary == {
    "Apple": 0,
    "Banana": 1,
    "Mango": 2
}

assert test_encoded == [0, 1, 0, 2, 1]

assert list(test_one_hot.columns) == [
    "Apple",
    "Banana",
    "Mango"
]

assert test_one_hot.values.tolist() == [
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
]

print("Label Encoding Unit Test: PASSED")
print("One-Hot Encoding Unit Test: PASSED")


# ------------------------------------------------------------
# A2 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A2 PERFORMANCE TEST ---")

large_data = ["A", "B", "C", "D", "E"] * 2000

start = time.perf_counter()

label_encoding(
    large_data,
    display=False
)

label_time = time.perf_counter() - start


start = time.perf_counter()

one_hot_encoding(
    large_data,
    display=False
)

one_hot_time = time.perf_counter() - start

print("Label Encoding Time:", label_time, "seconds")
print("One-Hot Encoding Time:", one_hot_time, "seconds")


# ============================================================
# A3
# Convert categorical features and study dimensionality
# ============================================================

print("\n\n")
print("############################################################")
print("A3 - CATEGORICAL FEATURE ENCODING")
print("############################################################")


categorical_features = [
    "Education",
    "Marital_Status"
]


# ------------------------------------------------------------
# ORIGINAL DATASET DIMENSIONALITY
# ------------------------------------------------------------

print("\nOriginal Dataset")

print("Number of Rows:", df.shape[0])
print("Number of Features:", df.shape[1])

print("\nOriginal Features:")
print(list(df.columns))


# ------------------------------------------------------------
# RECREATE DATASET AFTER ENCODING
# ------------------------------------------------------------

encoded_df = df.copy()

for column in categorical_features:

    values = encoded_df[column].fillna("Missing").tolist()

    temp_df = one_hot_encoding(
        values,
        display=False
    )

    # Rename encoded columns
    temp_df.columns = [
        column + "_" + str(value)
        for value in temp_df.columns
    ]

    # Remove original categorical column
    encoded_df.drop(
        column,
        axis=1,
        inplace=True
    )

    # Add encoded columns
    encoded_df = pd.concat(
        [
            encoded_df.reset_index(drop=True),
            temp_df.reset_index(drop=True)
        ],
        axis=1
    )


print("\nRecreated Dataset After Encoding")

print("Number of Rows:", encoded_df.shape[0])
print("Number of Features:", encoded_df.shape[1])

print("\nEncoded Dataset Features:")

for i, column in enumerate(
        encoded_df.columns,
        start=1
):
    print(i, ".", column)


print("\nFirst 5 Rows:")
print(encoded_df.head())


original_features = df.shape[1]
new_features = encoded_df.shape[1]

print("\nOriginal Number of Features:", original_features)
print("New Number of Features:", new_features)
print("Increase in Features:",
      new_features - original_features)


# ------------------------------------------------------------
# A3 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A3 FUNCTIONAL TESTING ---")

print("Original shape:", df.shape)
print("Encoded shape:", encoded_df.shape)

print("Categorical features converted successfully.")

for column in categorical_features:

    encoded_columns = [
        c for c in encoded_df.columns
        if c.startswith(column + "_")
    ]

    print(
        column,
        "->",
        len(encoded_columns),
        "encoded columns"
    )


# ------------------------------------------------------------
# A3 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A3 UNIT TESTS CONDUCTED ---")

assert encoded_df.shape[0] == df.shape[0]

for column in categorical_features:

    assert column not in encoded_df.columns

assert encoded_df.shape[1] >= df.shape[1]

print("Row Count Test: PASSED")
print("Original Categorical Columns Removed: PASSED")
print("Feature Dimensionality Test: PASSED")


# ------------------------------------------------------------
# A3 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A3 PERFORMANCE TEST ---")

start = time.perf_counter()

temp_encoded_df = df.copy()

for column in categorical_features:

    temp_values = temp_encoded_df[
        column
    ].fillna("Missing").tolist()

    temp_one_hot = one_hot_encoding(
        temp_values,
        display=False
    )

    temp_one_hot.columns = [
        column + "_" + str(value)
        for value in temp_one_hot.columns
    ]

    temp_encoded_df.drop(
        column,
        axis=1,
        inplace=True
    )

    temp_encoded_df = pd.concat(
        [
            temp_encoded_df.reset_index(drop=True),
            temp_one_hot.reset_index(drop=True)
        ],
        axis=1
    )

a3_time = time.perf_counter() - start

print("A3 Encoding Time:", a3_time, "seconds")


# ============================================================
# A4
# Minkowski Distance Function
# ============================================================

print("\n\n")
print("############################################################")
print("A4 - MINKOWSKI DISTANCE")
print("############################################################")


def minkowski(a, b, p):

    mink = 0

    for i in range(len(a)):

        if a[i] > b[i]:
            mink += (a[i] - b[i]) ** p

        else:
            mink += (b[i] - a[i]) ** p

    mink = math.pow(mink, 1 / p)

    return mink


a = [2, 4, 6]
b = [5, 8, 9]

print("\nExample Vectors:")
print("A =", a)
print("B =", b)

print("\nManhattan Distance:")

print(minkowski(a, b, 1))

print("\nEuclidean Distance:")

print(minkowski(a, b, 2))

print("\nMinkowski Distance for p = 3:")

print(minkowski(a, b, 3))


# ------------------------------------------------------------
# A4 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A4 FUNCTIONAL TESTING ---")

test_a = [2, 4]
test_b = [5, 8]

print("p = 1:", minkowski(test_a, test_b, 1))
print("p = 2:", minkowski(test_a, test_b, 2))


# ------------------------------------------------------------
# A4 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A4 UNIT TESTS CONDUCTED ---")

assert math.isclose(
    minkowski([2, 4], [5, 8], 1),
    7
)

assert math.isclose(
    minkowski([2, 4], [5, 8], 2),
    math.sqrt(25)
)

assert math.isclose(
    minkowski([1, 2, 3], [1, 2, 3], 2),
    0
)

print("Manhattan Distance Test: PASSED")
print("Euclidean Distance Test: PASSED")
print("Identical Vector Test: PASSED")


# ------------------------------------------------------------
# A4 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A4 PERFORMANCE TEST ---")

large_a = list(range(1000))
large_b = list(range(1000, 2000))

start = time.perf_counter()

for i in range(1000):
    minkowski(large_a, large_b, 2)

a4_time = time.perf_counter() - start

print(
    "1000 Minkowski Calculations Time:",
    a4_time,
    "seconds"
)


# ============================================================
# A5
# Minkowski Distance p = 1 to 10
# ============================================================

print("\n\n")
print("############################################################")
print("A5 - MINKOWSKI DISTANCE FOR p = 1 TO 10")
print("############################################################")


# Remove missing values only for required features

distance_df = df.dropna(
    subset=features
)

A = distance_df.iloc[0][
    features
].tolist()

B = distance_df.iloc[1][
    features
].tolist()


print("\nFeature Vector 1:")
print(A)

print("\nFeature Vector 2:")
print(B)


p_values = []
distances = []


for p in range(1, 11):

    distance = minkowski(A, B, p)

    p_values.append(p)
    distances.append(distance)

    print(
        "p =",
        p,
        "Distance =",
        distance
    )


# ------------------------------------------------------------
# A5 PLOT
# ------------------------------------------------------------

plt.plot(
    p_values,
    distances,
    marker='o'
)

plt.xlabel(
    "Order of Minkowski Distance (p)"
)

plt.ylabel(
    "Distance"
)

plt.title(
    "Minkowski Distance for p = 1 to 10"
)

plt.xticks(p_values)

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# A5 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A5 FUNCTIONAL TESTING ---")

assert len(p_values) == 10
assert len(distances) == 10

print("Distances calculated for p = 1 to 10.")
print("Graph generated successfully.")


# ------------------------------------------------------------
# A5 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A5 UNIT TESTS CONDUCTED ---")

small_A = [1, 2]
small_B = [4, 6]

test_distances = []

for p in range(1, 11):

    test_distances.append(
        minkowski(
            small_A,
            small_B,
            p
        )
    )

assert len(test_distances) == 10

assert math.isclose(
    test_distances[0],
    7
)

assert math.isclose(
    test_distances[1],
    5
)

print("p = 1 Test: PASSED")
print("p = 2 Test: PASSED")
print("p = 1 to 10 Test: PASSED")


# ------------------------------------------------------------
# A5 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A5 PERFORMANCE TEST ---")

start = time.perf_counter()

for p in range(1, 11):

    minkowski(
        A,
        B,
        p
    )

a5_time = time.perf_counter() - start

print(
    "10 Minkowski Distance Calculations Time:",
    a5_time,
    "seconds"
)


# ============================================================
# A6
# Compare with scipy.spatial.distance.minkowski()
# ============================================================

print("\n\n")
print("############################################################")
print("A6 - COMPARISON WITH SCIPY MINKOWSKI")
print("############################################################")


print("\np\tOur Function\tSciPy Function")

print("-----------------------------------------------")


scipy_results = []

for p in range(1, 11):

    my_distance = minkowski(
        A,
        B,
        p
    )

    scipy_distance = scipy_minkowski(
        A,
        B,
        p
    )

    scipy_results.append(
        scipy_distance
    )

    print(
        p,
        "\t",
        my_distance,
        "\t",
        scipy_distance
    )


print("\nResult Comparison")

for p in range(1, 11):

    my_distance = minkowski(
        A,
        B,
        p
    )

    scipy_distance = scipy_minkowski(
        A,
        B,
        p
    )

    if math.isclose(
        my_distance,
        scipy_distance
    ):
        print(
            "p =",
            p,
            ": Both results are SAME"
        )

    else:
        print(
            "p =",
            p,
            ": Results are DIFFERENT"
        )


# ------------------------------------------------------------
# A6 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A6 FUNCTIONAL TESTING ---")

test_A = [1, 2, 3]
test_B = [4, 6, 8]

for p in range(1, 11):

    own_result = minkowski(
        test_A,
        test_B,
        p
    )

    scipy_result = scipy_minkowski(
        test_A,
        test_B,
        p
    )

    print(
        "p =", p,
        "Own =", own_result,
        "SciPy =", scipy_result
    )


# ------------------------------------------------------------
# A6 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A6 UNIT TESTS CONDUCTED ---")

for p in range(1, 11):

    own_result = minkowski(
        test_A,
        test_B,
        p
    )

    scipy_result = scipy_minkowski(
        test_A,
        test_B,
        p
    )

    assert math.isclose(
        own_result,
        scipy_result
    )

print(
    "All p = 1 to 10 comparison tests: PASSED"
)


# ------------------------------------------------------------
# A6 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A6 PERFORMANCE TEST ---")

start = time.perf_counter()

for i in range(1000):

    scipy_minkowski(
        test_A,
        test_B,
        2
    )

a6_time = time.perf_counter() - start

print(
    "1000 SciPy Minkowski Calculations Time:",
    a6_time,
    "seconds"
)


# ============================================================
# A7
# Dot Product and Euclidean Norm
# ============================================================

print("\n\n")
print("############################################################")
print("A7 - DOT PRODUCT AND EUCLIDEAN NORM")
print("############################################################")


def dot_product(A, B):

    dot = 0

    for i in range(len(A)):

        dot += A[i] * B[i]

    return dot


def euclidean_norm(A):

    total = 0

    for i in range(len(A)):

        total += A[i] ** 2

    return math.sqrt(total)


A7 = [2, 4, 6, 8]
B7 = [1, 3, 5, 7]


my_dot = dot_product(
    A7,
    B7
)

numpy_dot = np.dot(
    A7,
    B7
)


my_norm_A = euclidean_norm(A7)
my_norm_B = euclidean_norm(B7)


numpy_norm_A = np.linalg.norm(A7)
numpy_norm_B = np.linalg.norm(B7)


print("\nVector A:", A7)
print("Vector B:", B7)

print("\nDot Product")

print(
    "Our Dot Product:",
    my_dot
)

print(
    "NumPy Dot Product:",
    numpy_dot
)


print("\nEuclidean Norm")

print(
    "Our Norm of A:",
    my_norm_A
)

print(
    "NumPy Norm of A:",
    numpy_norm_A
)

print(
    "Our Norm of B:",
    my_norm_B
)

print(
    "NumPy Norm of B:",
    numpy_norm_B
)


# ------------------------------------------------------------
# A7 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A7 FUNCTIONAL TESTING ---")

print(
    "Dot Product:",
    dot_product(A7, B7)
)

print(
    "Euclidean Norm A:",
    euclidean_norm(A7)
)

print(
    "Euclidean Norm B:",
    euclidean_norm(B7)
)


# ------------------------------------------------------------
# A7 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A7 UNIT TESTS CONDUCTED ---")

assert dot_product(
    [1, 2, 3],
    [4, 5, 6]
) == 32

assert math.isclose(
    euclidean_norm([3, 4]),
    5
)

assert math.isclose(
    euclidean_norm([0, 0]),
    0
)

assert dot_product(
    A7,
    B7
) == np.dot(
    A7,
    B7
)

assert math.isclose(
    euclidean_norm(A7),
    np.linalg.norm(A7)
)

print("Dot Product Test: PASSED")
print("Euclidean Norm Test: PASSED")
print("Zero Vector Test: PASSED")
print("NumPy Comparison Test: PASSED")


# ------------------------------------------------------------
# A7 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A7 PERFORMANCE TEST ---")

large_A = list(range(10000))
large_B = list(range(10000))

start = time.perf_counter()

for i in range(100):

    dot_product(
        large_A,
        large_B
    )

dot_time = time.perf_counter() - start


start = time.perf_counter()

for i in range(100):

    euclidean_norm(
        large_A
    )

norm_time = time.perf_counter() - start


print(
    "100 Dot Product Calculations:",
    dot_time,
    "seconds"
)

print(
    "100 Euclidean Norm Calculations:",
    norm_time,
    "seconds"
)


# ============================================================
# A8
# Mean, Variance and Standard Deviation
# ============================================================

print("\n\n")
print("############################################################")
print("A8 - MEAN, VARIANCE AND STANDARD DEVIATION")
print("############################################################")


def mean(data):

    total = 0

    for value in data:

        total += value

    return total / len(data)


def variance(data):

    avg = mean(data)

    total = 0

    for value in data:

        total += (
            value - avg
        ) ** 2

    return total / len(data)


def standard_deviation(data):

    var = variance(data)

    return math.sqrt(var)


def calculate_statistics(dataset):

    print("\nFeature\t\tMean\t\tVariance\tStd Deviation")

    print(
        "---------------------------------------------------------------"
    )

    for column in dataset.columns:

        if pd.api.types.is_numeric_dtype(
            dataset[column]
        ):

            data = dataset[
                column
            ].dropna().tolist()

            m = mean(data)
            v = variance(data)
            sd = standard_deviation(data)

            print(
                f"{column:20}"
                f"{m:12.2f}"
                f"{v:18.2f}"
                f"{sd:18.2f}"
            )


calculate_statistics(df)


# ------------------------------------------------------------
# A8 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A8 FUNCTIONAL TESTING ---")

test_data = [2, 4, 6, 8]

print(
    "Data:",
    test_data
)

print(
    "Mean:",
    mean(test_data)
)

print(
    "Variance:",
    variance(test_data)
)

print(
    "Standard Deviation:",
    standard_deviation(test_data)
)


# ------------------------------------------------------------
# A8 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A8 UNIT TESTS CONDUCTED ---")

assert math.isclose(
    mean([2, 4, 6, 8]),
    5
)

assert math.isclose(
    variance([2, 4, 6, 8]),
    5
)

assert math.isclose(
    standard_deviation([2, 4, 6, 8]),
    math.sqrt(5)
)

assert mean([5, 5, 5]) == 5

assert variance([5, 5, 5]) == 0

print("Mean Test: PASSED")
print("Variance Test: PASSED")
print("Standard Deviation Test: PASSED")
print("Constant Data Test: PASSED")


# ------------------------------------------------------------
# A8 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A8 PERFORMANCE TEST ---")

large_data = list(range(10000))

start = time.perf_counter()

for i in range(100):

    mean(large_data)

performance_mean_time = (
    time.perf_counter() - start
)


start = time.perf_counter()

for i in range(100):

    variance(large_data)

performance_variance_time = (
    time.perf_counter() - start
)


start = time.perf_counter()

for i in range(100):

    standard_deviation(
        large_data
    )

performance_std_time = (
    time.perf_counter() - start
)


print(
    "100 Mean Calculations:",
    performance_mean_time,
    "seconds"
)

print(
    "100 Variance Calculations:",
    performance_variance_time,
    "seconds"
)

print(
    "100 Standard Deviation Calculations:",
    performance_std_time,
    "seconds"
)


# ============================================================
# A9
# Compare A8 Functions with NumPy
# ============================================================

print("\n\n")
print("############################################################")
print("A9 - COMPARISON WITH NUMPY")
print("############################################################")


numerical_df = df.select_dtypes(
    include=np.number
)

numerical_df = numerical_df.dropna()

feat_vecs = numerical_df.values


my_mean = []
my_std = []


for column in numerical_df.columns:

    data = numerical_df[
        column
    ].tolist()

    my_mean.append(
        mean(data)
    )

    my_std.append(
        standard_deviation(data)
    )


numpy_mean = np.mean(
    feat_vecs,
    axis=0
)

numpy_std = np.std(
    feat_vecs,
    axis=0
)


print(
    "\nFeature\t\tOur Mean\tNumPy Mean\tOur Std\tNumPy Std"
)

print(
    "--------------------------------------------------------------------------"
)


for i in range(
    len(numerical_df.columns)
):

    print(
        f"{numerical_df.columns[i]:20}"
        f"{my_mean[i]:12.2f}"
        f"{numpy_mean[i]:15.2f}"
        f"{my_std[i]:15.2f}"
        f"{numpy_std[i]:15.2f}"
    )


print("\nComparison")

if np.allclose(
    my_mean,
    numpy_mean
):

    print(
        "Mean vectors are SAME"
    )

else:

    print(
        "Mean vectors are DIFFERENT"
    )


if np.allclose(
    my_std,
    numpy_std
):

    print(
        "Standard deviation vectors are SAME"
    )

else:

    print(
        "Standard deviation vectors are DIFFERENT"
    )


# ------------------------------------------------------------
# A9 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A9 FUNCTIONAL TESTING ---")

test_matrix = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

test_my_mean = [
    mean(test_matrix[:, 0].tolist()),
    mean(test_matrix[:, 1].tolist())
]

test_my_std = [
    standard_deviation(
        test_matrix[:, 0].tolist()
    ),
    standard_deviation(
        test_matrix[:, 1].tolist()
    )
]

print(
    "Our Mean Vector:",
    test_my_mean
)

print(
    "NumPy Mean Vector:",
    np.mean(
        test_matrix,
        axis=0
    )
)

print(
    "Our Std Vector:",
    test_my_std
)

print(
    "NumPy Std Vector:",
    np.std(
        test_matrix,
        axis=0
    )
)


# ------------------------------------------------------------
# A9 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A9 UNIT TESTS CONDUCTED ---")

assert np.allclose(
    test_my_mean,
    np.mean(
        test_matrix,
        axis=0
    )
)

assert np.allclose(
    test_my_std,
    np.std(
        test_matrix,
        axis=0
    )
)

print("Mean Vector Comparison: PASSED")
print("Standard Deviation Vector Comparison: PASSED")


# ------------------------------------------------------------
# A9 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A9 PERFORMANCE TEST ---")

start = time.perf_counter()

for i in range(100):

    np.mean(
        feat_vecs,
        axis=0
    )

numpy_mean_time = (
    time.perf_counter() - start
)


start = time.perf_counter()

for i in range(100):

    np.std(
        feat_vecs,
        axis=0
    )

numpy_std_time = (
    time.perf_counter() - start
)


print(
    "100 NumPy Mean Calculations:",
    numpy_mean_time,
    "seconds"
)

print(
    "100 NumPy Std Calculations:",
    numpy_std_time,
    "seconds"
)


# ============================================================
# A10
# Histogram, Mean and Variance
# ============================================================

print("\n\n")
print("############################################################")
print("A10 - HISTOGRAM, MEAN AND VARIANCE")
print("############################################################")


feature = "Income"

data = df[
    feature
].dropna().values


feature_mean = mean(data)

feature_variance = variance(data)


print("\nFeature:", feature)

print(
    "Mean:",
    feature_mean
)

print(
    "Variance:",
    feature_variance
)


hist, bins = np.histogram(
    data,
    bins=10
)


print("\nBuckets:")

print(bins)


print("\nFrequency:")

print(hist)


# ------------------------------------------------------------
# PLOT HISTOGRAM
# ------------------------------------------------------------

plt.hist(
    data,
    bins=10
)

plt.xlabel(
    "Income"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Histogram / Density Pattern of Income"
)

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# A10 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A10 FUNCTIONAL TESTING ---")

test_data = np.array([
    10, 20, 20, 30, 30,
    30, 40, 50, 50, 60
])

test_hist, test_bins = np.histogram(
    test_data,
    bins=5
)

print(
    "Histogram Frequencies:",
    test_hist
)

print(
    "Histogram Buckets:",
    test_bins
)

print(
    "Mean:",
    mean(test_data)
)

print(
    "Variance:",
    variance(test_data)
)


# ------------------------------------------------------------
# A10 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A10 UNIT TESTS CONDUCTED ---")

assert len(test_hist) == 5

assert len(test_bins) == 6

assert sum(test_hist) == len(test_data)

assert math.isclose(
    mean(test_data),
    np.mean(test_data)
)

assert math.isclose(
    variance(test_data),
    np.var(test_data)
)

print("Histogram Bucket Test: PASSED")
print("Histogram Frequency Test: PASSED")
print("Mean Test: PASSED")
print("Variance Test: PASSED")


# ------------------------------------------------------------
# A10 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A10 PERFORMANCE TEST ---")

large_hist_data = np.tile(
    data,
    5
)

start = time.perf_counter()

for i in range(100):

    np.histogram(
        large_hist_data,
        bins=10
    )

a10_time = (
    time.perf_counter() - start
)

print(
    "100 Histogram Calculations:",
    a10_time,
    "seconds"
)


# ============================================================
# A11
# K-MEANS ALGORITHM
# ============================================================

print("\n\n")
print("############################################################")
print("A11 - K-MEANS ALGORITHM")
print("############################################################")


# ------------------------------------------------------------
# CENTROID CALCULATION
# ------------------------------------------------------------

def calculate_centroid(cluster):

    if len(cluster) == 0:

        return None

    centroid = []

    for i in range(
        len(cluster[0])
    ):

        values = []

        for point in cluster:

            values.append(
                point[i]
            )

        centroid.append(
            mean(values)
        )

    return centroid


# ------------------------------------------------------------
# K-MEANS FUNCTION
# ------------------------------------------------------------

def k_means(
    data,
    k,
    max_iterations=100
):

    # --------------------------------------------------------
    # STEP 1:
    # Select K initial centroids
    # --------------------------------------------------------

    centroids = []

    for i in range(k):

        centroids.append(
            data[i].copy()
        )


    iteration = 0


    # --------------------------------------------------------
    # REPEAT
    # --------------------------------------------------------

    while iteration < max_iterations:

        iteration += 1


        # ----------------------------------------------------
        # STEP 2:
        # Create K empty clusters
        # ----------------------------------------------------

        clusters = []

        for i in range(k):

            clusters.append([])


        # ----------------------------------------------------
        # STEP 3:
        # Assign each point to closest centroid
        # ----------------------------------------------------

        for point in data:

            distances = []

            for centroid in centroids:

                distance = minkowski(
                    point,
                    centroid,
                    2
                )

                distances.append(
                    distance
                )


            closest_cluster = (
                distances.index(
                    min(distances)
                )
            )


            clusters[
                closest_cluster
            ].append(point)


        # ----------------------------------------------------
        # STEP 4:
        # Recalculate centroids
        # ----------------------------------------------------

        new_centroids = []


        for i in range(k):

            cluster = clusters[i]


            if len(cluster) == 0:

                new_centroids.append(
                    centroids[i].copy()
                )

            else:

                new_centroids.append(
                    calculate_centroid(
                        cluster
                    )
                )


        # ----------------------------------------------------
        # STEP 5:
        # Check if centroids changed
        # ----------------------------------------------------

        if np.allclose(
            centroids,
            new_centroids
        ):

            break


        centroids = new_centroids


    return centroids, clusters, iteration


# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------

kmeans_df = df[
    features
].dropna()


# Use a smaller set of features for easier clustering
# and visualization/processing.

kmeans_features = [
    "Income",
    "MntWines",
    "MntMeatProducts"
]


kmeans_df = df[
    kmeans_features
].dropna()


kmeans_data = (
    kmeans_df.values.tolist()
)


# ------------------------------------------------------------
# RUN K-MEANS
# ------------------------------------------------------------

k = 3


final_centroids, final_clusters, iterations = k_means(
    kmeans_data,
    k
)


print("\nInitial K =", k)

print(
    "Number of Iterations:",
    iterations
)


print("\nFinal Centroids")

for i in range(k):

    print(
        "Centroid",
        i + 1,
        ":",
        final_centroids[i]
    )


print("\nCluster Sizes")

for i in range(k):

    print(
        "Cluster",
        i + 1,
        ":",
        len(final_clusters[i]),
        "points"
    )


# ------------------------------------------------------------
# A11 FUNCTIONAL TESTING
# ------------------------------------------------------------

print("\n--- A11 FUNCTIONAL TESTING ---")


test_kmeans_data = [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],

    [10, 10],
    [10, 11],
    [11, 10],
    [11, 11]
]


test_centroids, test_clusters, test_iterations = k_means(
    test_kmeans_data,
    2
)


print(
    "Final Centroids:",
    test_centroids
)

print(
    "Cluster Sizes:",
    [
        len(cluster)
        for cluster in test_clusters
    ]
)

print(
    "Iterations:",
    test_iterations
)


# ------------------------------------------------------------
# A11 UNIT TESTS
# ------------------------------------------------------------

print("\n--- A11 UNIT TESTS CONDUCTED ---")

assert len(test_centroids) == 2

assert len(test_clusters) == 2

assert sum(
    len(cluster)
    for cluster in test_clusters
) == len(test_kmeans_data)


assert test_iterations > 0

print("Number of Centroids Test: PASSED")
print("Number of Clusters Test: PASSED")
print("All Points Assigned Test: PASSED")
print("Iteration Test: PASSED")


# ------------------------------------------------------------
# A11 PERFORMANCE TEST
# ------------------------------------------------------------

print("\n--- A11 PERFORMANCE TEST ---")

performance_data = []

for i in range(200):

    performance_data.append([
        i % 50,
        (i * 2) % 100,
        (i * 3) % 150
    ])


start = time.perf_counter()

k_means(
    performance_data,
    3
)

a11_time = (
    time.perf_counter() - start
)


print(
    "K-Means on 200 points:",
    a11_time,
    "seconds"
)


# ============================================================
# FINAL TESTING SUMMARY
# ============================================================

print("\n\n")
print("############################################################")
print("FINAL TESTING SUMMARY")
print("############################################################")

print("""
A2  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A3  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A4  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A5  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A6  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A7  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A8  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A9  -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A10 -> Functional Testing + Unit Tests + Performance Test : COMPLETED
A11 -> Functional Testing + Unit Tests + Performance Test : COMPLETED
""")

print("All unit tests completed successfully.")