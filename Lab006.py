# ================================================================
# LAB 06 - K NEAREST NEIGHBOUR CLASSIFIER
# DATASET : MARKETING CAMPAIGN
# A1 TO A9
# ================================================================


# ================================================================
# IMPORT LIBRARIES
# ================================================================

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ================================================================
# LOAD DATASET
# ================================================================
df = pd.read_excel(
    r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab04.xlsx",
    sheet_name="marketing_campaign"
)

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)


# ================================================================
# A1 : MODULAR KNN CLASSIFIER
# ================================================================


# ------------------------------------------------
# MEAN
# ------------------------------------------------

def mean(numbers):

    length = len(numbers)
    total = 0

    for i in range(len(numbers)):
        total += numbers[i]

    avg = total / length

    return avg


# ------------------------------------------------
# MEDIAN
# ------------------------------------------------

def median(numbers):

    numbers = numbers.copy()
    numbers.sort()

    n = len(numbers)

    if n % 2 != 0:

        return numbers[n // 2]

    else:

        return (
            numbers[(n // 2) - 1] +
            numbers[n // 2]
        ) / 2


# ------------------------------------------------
# MODE
# ------------------------------------------------

def mode(numbers):

    max_count = 0
    mode_value = None

    for num in numbers:

        count = 0

        for x in numbers:

            if x == num:
                count += 1

        if count > max_count:

            max_count = count
            mode_value = num

    return mode_value, max_count


# ------------------------------------------------
# FIND CENTRAL TENDENCIES
# ------------------------------------------------

def find_central_tendencies(numbers):

    mean_value = mean(numbers)
    median_value = median(numbers)
    mode_value, mode_count = mode(numbers)

    return (
        mean_value,
        median_value,
        mode_value
    )


# ------------------------------------------------
# DATA IMPUTATION
# ------------------------------------------------

def impute_data(data, method="median"):

    data = data.copy()

    for column in data.columns:

        values = data[column].dropna().tolist()

        if len(values) == 0:
            continue

        mean_value, median_value, mode_value = (
            find_central_tendencies(values)
        )

        if method == "mean":

            fill_value = mean_value

        elif method == "median":

            fill_value = median_value

        elif method == "mode":

            fill_value = mode_value

        else:

            fill_value = median_value

        data[column] = data[column].fillna(
            fill_value
        )

    return data


# ------------------------------------------------
# EUCLIDEAN DISTANCE
# ------------------------------------------------

def euclidean(a, b):

    euc = 0

    for i in range(len(a)):

        if a[i] > b[i]:

            euc += (a[i] - b[i]) ** 2

        else:

            euc += (b[i] - a[i]) ** 2

    euc = math.sqrt(euc)

    return euc


# ------------------------------------------------
# BUBBLE SORT
# ------------------------------------------------

def bubble_sort(data):

    data = data.copy()

    n = len(data)

    for i in range(n):

        for j in range(0, n - i - 1):

            if data[j][0] > data[j + 1][0]:

                data[j], data[j + 1] = (
                    data[j + 1],
                    data[j]
                )

    return data


# ------------------------------------------------
# SELECTION SORT
# ------------------------------------------------

def selection_sort(data):

    data = data.copy()

    n = len(data)

    for i in range(n):

        min_index = i

        for j in range(i + 1, n):

            if data[j][0] < data[min_index][0]:

                min_index = j

        data[i], data[min_index] = (
            data[min_index],
            data[i]
        )

    return data


# ------------------------------------------------
# INSERTION SORT
# ------------------------------------------------

def insertion_sort(data):

    data = data.copy()

    for i in range(1, len(data)):

        key = data[i]

        j = i - 1

        while (
            j >= 0 and
            data[j][0] > key[0]
        ):

            data[j + 1] = data[j]

            j -= 1

        data[j + 1] = key

    return data


# ------------------------------------------------
# SORTING CONFIGURATION
# ------------------------------------------------

def sort_data(data, algorithm="bubble"):

    if algorithm == "bubble":

        return bubble_sort(data)

    elif algorithm == "selection":

        return selection_sort(data)

    elif algorithm == "insertion":

        return insertion_sort(data)

    else:

        return bubble_sort(data)


# ------------------------------------------------
# IDENTIFY NEIGHBOURS
# ------------------------------------------------

def identify_neighbors(
        X_train,
        y_train,
        test_point,
        k,
        sorting_algorithm="bubble"):

    distances = []

    for i in range(len(X_train)):

        distance = euclidean(
            test_point,
            X_train[i]
        )

        distances.append(
            (
                distance,
                y_train[i],
                i
            )
        )

    distances = sort_data(
        distances,
        sorting_algorithm
    )

    return distances[:k]


# ------------------------------------------------
# FAST NEIGHBOUR IDENTIFICATION
#
# Used by A7-A9 to avoid repeatedly doing
# expensive full bubble/selection/insertion sorts.
#
# A1 still contains all three required DSA
# sorting algorithms.
# ------------------------------------------------

def fast_identify_neighbors(
        X_train,
        y_train,
        test_point,
        k):

    nearest = []

    for i in range(len(X_train)):

        distance = euclidean(
            test_point,
            X_train[i]
        )

        current = (
            distance,
            y_train[i],
            i
        )

        # Insert current point in the correct
        # position among the k nearest points.

        position = len(nearest)

        for j in range(len(nearest)):

            if distance < nearest[j][0]:

                position = j
                break

        nearest.insert(
            position,
            current
        )

        # Keep only k neighbours
        if len(nearest) > k:

            nearest.pop()

    return nearest


# ------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------

def classify(neighbors):

    class_counts = {}

    for neighbor in neighbors:

        label = neighbor[1]

        if label not in class_counts:

            class_counts[label] = 0

        class_counts[label] += 1

    max_count = max(
        class_counts.values()
    )

    candidates = []

    for label in class_counts:

        if class_counts[label] == max_count:

            candidates.append(label)

    # Tie breaking:
    # choose the smallest class label

    return min(candidates)


# ------------------------------------------------
# NORMAL KNN
# ------------------------------------------------

def knn_classify(
        X_train,
        y_train,
        test_point,
        k,
        sorting_algorithm="bubble"):

    neighbors = identify_neighbors(
        X_train,
        y_train,
        test_point,
        k,
        sorting_algorithm
    )

    prediction = classify(
        neighbors
    )

    return prediction


# ================================================================
# PREPARE DATA FOR A1
# ================================================================


# ------------------------------------------------
# SELECT NUMERICAL FEATURES
# ------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# ------------------------------------------------
# REMOVE ID AND TARGET FROM FEATURES
# ------------------------------------------------

if "ID" in numeric_columns:

    numeric_columns.remove("ID")


if "Response" in numeric_columns:

    numeric_columns.remove("Response")


print("\nNumerical Features Used:")

for column in numeric_columns:

    print(column)


# ------------------------------------------------
# FIND CENTRAL TENDENCIES
# ------------------------------------------------

print("\nCENTRAL TENDENCIES")

for column in numeric_columns:

    values = df[column].dropna().tolist()

    mean_value, median_value, mode_value = (
        find_central_tendencies(values)
    )

    print("\n", column)
    print("Mean   :", mean_value)
    print("Median :", median_value)
    print("Mode   :", mode_value)


# ------------------------------------------------
# IMPUTE MISSING VALUES
# ------------------------------------------------

# Median is used as the selected imputation method

df[numeric_columns] = impute_data(
    df[numeric_columns],
    method="median"
)


# ------------------------------------------------
# CREATE X AND Y
# ------------------------------------------------

X = df[numeric_columns].values
y = df["Response"].values


# Convert NumPy arrays into lists

X = X.tolist()
y = y.tolist()


print("\nX Shape :", len(X))
print("Y Shape :", len(y))


# ================================================================
# A1 : TEST MANUAL KNN
# ================================================================

print("\n================ A1 ================")

test_point = X[0]

neighbors = identify_neighbors(
    X,
    y,
    test_point,
    k=3,
    sorting_algorithm="bubble"
)

print("\n3 Nearest Neighbours:")

for neighbor in neighbors:

    print(
        "Distance :",
        neighbor[0],
        "Class :",
        neighbor[1]
    )

prediction = classify(
    neighbors
)

print("\nPredicted Class :", prediction)


# ================================================================
# A2 : WEIGHTED KNN
# ================================================================


# ------------------------------------------------
# WEIGHTED CLASSIFICATION
# ------------------------------------------------

def weighted_classify(neighbors):

    class_weights = {}

    for neighbor in neighbors:

        distance = neighbor[0]
        label = neighbor[1]

        # If distance is zero, the point is
        # exactly the same as the test point.

        if distance == 0:

            return label

        weight = 1 / distance

        if label not in class_weights:

            class_weights[label] = 0

        class_weights[label] += weight


    # Find maximum class weight

    max_weight = max(
        class_weights.values()
    )

    candidates = []

    for label in class_weights:

        if class_weights[label] == max_weight:

            candidates.append(label)


    # Tie breaking

    return min(candidates)


# ------------------------------------------------
# WEIGHTED KNN
# ------------------------------------------------

def weighted_knn(
        X_train,
        y_train,
        test_point,
        k,
        sorting_algorithm="bubble"):

    # Reuse A1 neighbour identification

    neighbors = identify_neighbors(
        X_train,
        y_train,
        test_point,
        k,
        sorting_algorithm
    )

    # Use weighted classification

    prediction = weighted_classify(
        neighbors
    )

    return prediction


# ------------------------------------------------
# A2 TEST
# ------------------------------------------------

print("\n================ A2 ================")

weighted_prediction = weighted_knn(
    X,
    y,
    test_point,
    k=3,
    sorting_algorithm="bubble"
)

print(
    "Weighted KNN Prediction :",
    weighted_prediction
)


# ================================================================
# A3 : TRAIN TEST SPLIT
# ================================================================

print("\n================ A3 ================")

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.3
    )
)


print(
    "Total samples    :",
    len(X)
)

print(
    "Training samples :",
    len(X_train)
)

print(
    "Testing samples  :",
    len(X_test)
)


# ================================================================
# A4 : PACKAGED KNN CLASSIFIER
# ================================================================

print("\n================ A4 ================")


neigh = KNeighborsClassifier(
    n_neighbors=3
)


neigh.fit(
    X_train,
    y_train
)


print("K value :", 3)

print(
    "Training completed successfully"
)


# ================================================================
# A5 : NEIGH SCORE
# ================================================================

print("\n================ A5 ================")


neighscore = neigh.score(
    X_test,
    y_test
)


print(
    "Score :",
    neighscore
)

print(
    "Accuracy :",
    neighscore * 100,
    "%"
)


# ================================================================
# A6 : NEIGH PREDICT
# ================================================================

print("\n================ A6 ================")


neighpredict = neigh.predict(
    X_test
)


print(
    "Predicted Classes:"
)

print(
    neighpredict.tolist()
)


# ================================================================
# A7 : MANUAL FIT()
# ================================================================


def fit(X_train, y_train):

    model = {}

    model["X_train"] = X_train

    model["y_train"] = y_train

    return model


# ================================================================
# A7 : MANUAL PREDICT()
# ================================================================

def predict(
        model,
        X_test,
        k=3):

    X_train = model["X_train"]

    y_train = model["y_train"]

    predictions = []


    for test_point in X_test:

        # Use FAST neighbour identification
        # instead of bubble sort.
        #
        # This prevents A7/A8 from becoming
        # extremely slow.

        neighbors = fast_identify_neighbors(
            X_train,
            y_train,
            test_point,
            k
        )


        # Reuse A1 majority voting

        prediction = classify(
            neighbors
        )


        predictions.append(
            prediction
        )


    return predictions


# ================================================================
# A7 : MANUAL SCORE()
# ================================================================

def score(
        model,
        X_test,
        y_test,
        k=3):

    predictions = predict(
        model,
        X_test,
        k
    )

    correct = 0


    for i in range(len(y_test)):

        if predictions[i] == y_test[i]:

            correct += 1


    accuracy = (
        correct /
        len(y_test)
    )


    return accuracy


# ================================================================
# A7 : EXECUTION
# ================================================================

print("\n================ A7 ================")


my_knn = fit(
    X_train,
    y_train
)


mypredict = predict(
    my_knn,
    X_test,
    k=3
)


print(
    "\nA7 - PREDICT"
)

print(
    "Predicted Classes:"
)

print(
    mypredict
)


myscore = score(
    my_knn,
    X_test,
    y_test,
    k=3
)


print(
    "\nA7 - SCORE"
)

print(
    "Score :",
    myscore
)

print(
    "Accuracy :",
    myscore * 100,
    "%"
)


# ================================================================
# A8 : COMPARE PACKAGED KNN AND CREATED KNN
# ================================================================

print("\n================ A8 ================")


# Test different values of K

k_values = range(
    1,
    16
)


packaged_accuracy = []

created_accuracy = []


# Fit manual model only once

my_knn = fit(
    X_train,
    y_train
)


for k in k_values:


    # ------------------------------------------------
    # PACKAGED KNN
    # ------------------------------------------------

    neigh = KNeighborsClassifier(
        n_neighbors=k
    )


    neigh.fit(
        X_train,
        y_train
    )


    score1 = neigh.score(
        X_test,
        y_test
    )


    packaged_accuracy.append(
        score1
    )


    # ------------------------------------------------
    # CREATED KNN
    # ------------------------------------------------

    score2 = score(
        my_knn,
        X_test,
        y_test,
        k=k
    )


    created_accuracy.append(
        score2
    )


# ------------------------------------------------
# DISPLAY A8 RESULTS
# ------------------------------------------------

print(
    "\nK\tPackaged KNN\tCreated KNN"
)


for i in range(
        len(k_values)):

    print(
        k_values[i],
        "\t",
        packaged_accuracy[i],
        "\t",
        created_accuracy[i]
    )


# ------------------------------------------------
# A8 GRAPH
# ------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    list(k_values),
    packaged_accuracy,
    marker="o",
    label="Packaged KNN"
)


plt.plot(
    list(k_values),
    created_accuracy,
    marker="s",
    label="Created KNN"
)


plt.xlabel(
    "Value of K"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Packaged KNN vs Created KNN"
)


plt.xticks(
    list(k_values)
)


plt.legend()


plt.grid()


plt.show()


# ================================================================
# A9 : WEIGHTED KNN
# ================================================================

print("\n================ A9 ================")


weighted_accuracy = []


for k in k_values:

    correct = 0


    for i in range(
            len(X_test)):


        # ------------------------------------------------
        # REUSE A2 WEIGHTED KNN
        #
        # We use the FAST neighbour identification
        # so the experiment does not get stuck.
        # ------------------------------------------------

        neighbors = fast_identify_neighbors(
            X_train,
            y_train,
            X_test[i],
            k
        )


        prediction = weighted_classify(
            neighbors
        )


        if prediction == y_test[i]:

            correct += 1


    accuracy = (
        correct /
        len(y_test)
    )


    weighted_accuracy.append(
        accuracy
    )


# ------------------------------------------------
# DISPLAY A9 RESULTS
# ------------------------------------------------

print(
    "\nK\tPackaged KNN\tCreated KNN\tWeighted KNN"
)


for i in range(
        len(k_values)):

    print(
        k_values[i],
        "\t",
        packaged_accuracy[i],
        "\t",
        created_accuracy[i],
        "\t",
        weighted_accuracy[i]
    )


# ------------------------------------------------
# A9 GRAPH
# ------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    list(k_values),
    packaged_accuracy,
    marker="o",
    label="Packaged KNN"
)


plt.plot(
    list(k_values),
    created_accuracy,
    marker="s",
    label="Created KNN"
)


plt.plot(
    list(k_values),
    weighted_accuracy,
    marker="^",
    label="Weighted KNN"
)


plt.xlabel(
    "Value of K"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Packaged KNN vs Created KNN vs Weighted KNN"
)


plt.xticks(
    list(k_values)
)


plt.legend()


plt.grid()


plt.show()


# ================================================================
# FINAL BEST K VALUES
# ================================================================

print("\n================ FINAL RESULTS ================")


best_packaged_index = packaged_accuracy.index(
    max(packaged_accuracy)
)


best_created_index = created_accuracy.index(
    max(created_accuracy)
)


best_weighted_index = weighted_accuracy.index(
    max(weighted_accuracy)
)


print(
    "Best Packaged KNN K :",
    list(k_values)[best_packaged_index]
)

print(
    "Best Packaged Accuracy :",
    packaged_accuracy[best_packaged_index] * 100,
    "%"
)


print(
    "\nBest Created KNN K :",
    list(k_values)[best_created_index]
)

print(
    "Best Created Accuracy :",
    created_accuracy[best_created_index] * 100,
    "%"
)


print(
    "\nBest Weighted KNN K :",
    list(k_values)[best_weighted_index]
)

print(
    "Best Weighted Accuracy :",
    weighted_accuracy[best_weighted_index] * 100,
    "%"
)