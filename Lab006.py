import pandas as pd
import math

# ==================================================
# A1 : K-NEAREST NEIGHBOURS
# ==================================================

# Read Excel file
df = pd.read_excel(
    r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\Lab_Session_Data_Lab04.xlsx",
    sheet_name="marketing_campaign"
)


# --------------------------------------------------
# MEAN, MEDIAN, MODE
# --------------------------------------------------

def mean(numbers):
    length = len(numbers)
    sum = 0

    for i in range(len(numbers)):
        sum += numbers[i]

    avg = sum / length
    return avg


def median(numbers):
    numbers = numbers.copy()
    numbers.sort()
    n = len(numbers)

    if n % 2 != 0:
        return numbers[n // 2]

    else:
        return (numbers[(n // 2) - 1] + numbers[n // 2]) / 2


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


# --------------------------------------------------
# DATA IMPUTATION
# --------------------------------------------------

def impute_data(df, method="mean"):

    df = df.copy()

    for column in df.columns:

        if df[column].isnull().sum() > 0:

            values = df[column].dropna().tolist()

            if method == "mean":
                value = mean(values)

            elif method == "median":
                value = median(values)

            elif method == "mode":
                value, count = mode(values)

            df[column] = df[column].fillna(value)

    return df


# --------------------------------------------------
# EUCLIDEAN DISTANCE
# --------------------------------------------------

def euclidean(a, b):

    euc = 0

    for i in range(len(a)):

        if a[i] > b[i]:
            euc += (a[i] - b[i]) ** 2

        else:
            euc += (b[i] - a[i]) ** 2

    euc = math.sqrt(euc)

    return euc


# --------------------------------------------------
# BUBBLE SORT
# --------------------------------------------------

def bubble_sort(data):

    data = data.copy()
    n = len(data)

    for i in range(n):

        for j in range(0, n - i - 1):

            if data[j][0] > data[j + 1][0]:

                data[j], data[j + 1] = data[j + 1], data[j]

    return data


# --------------------------------------------------
# SELECTION SORT
# --------------------------------------------------

def selection_sort(data):

    data = data.copy()
    n = len(data)

    for i in range(n):

        min_index = i

        for j in range(i + 1, n):

            if data[j][0] < data[min_index][0]:
                min_index = j

        data[i], data[min_index] = data[min_index], data[i]

    return data


# --------------------------------------------------
# INSERTION SORT
# --------------------------------------------------

def insertion_sort(data):

    data = data.copy()

    for i in range(1, len(data)):

        key = data[i]
        j = i - 1

        while j >= 0 and data[j][0] > key[0]:

            data[j + 1] = data[j]
            j -= 1

        data[j + 1] = key

    return data


# --------------------------------------------------
# FIND K NEAREST NEIGHBOURS
# --------------------------------------------------

def identify_neighbors(X_train, y_train, test_point, k,
                       sorting_algorithm="bubble"):

    distances = []

    for i in range(len(X_train)):

        distance = euclidean(test_point, X_train[i])

        distances.append((distance, y_train[i], i))

    if sorting_algorithm == "bubble":
        distances = bubble_sort(distances)

    elif sorting_algorithm == "selection":
        distances = selection_sort(distances)

    elif sorting_algorithm == "insertion":
        distances = insertion_sort(distances)

    else:
        print("Invalid sorting algorithm")
        return []

    return distances[:k]


# --------------------------------------------------
# CLASS ASSIGNMENT
# --------------------------------------------------

def classify(neighbors):

    votes = {}

    for distance, label, index in neighbors:

        if label not in votes:
            votes[label] = 0

        votes[label] += 1

    max_votes = max(votes.values())

    winners = []

    for label in votes:

        if votes[label] == max_votes:
            winners.append(label)

    # Tie breaking:
    # choose the class of the closest neighbour

    if len(winners) > 1:

        for distance, label, index in neighbors:

            if label in winners:
                return label

    return winners[0]


# --------------------------------------------------
# KNN CLASSIFIER
# --------------------------------------------------

def knn(X_train, y_train, X_test, k=5,
        sorting_algorithm="bubble"):

    predictions = []

    for test_point in X_test:

        neighbors = identify_neighbors(
            X_train,
            y_train,
            test_point,
            k,
            sorting_algorithm
        )

        prediction = classify(neighbors)

        predictions.append(prediction)

    return predictions


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

numeric_features = []

for column in numeric_columns:

    if column != "Response":
        numeric_features.append(column)

print("Numerical Features:")
print(numeric_features)


# --------------------------------------------------
# FIND MEAN, MEDIAN AND MODE
# --------------------------------------------------

print("\nCENTRAL TENDENCIES")

for column in numeric_features:

    values = df[column].dropna().tolist()

    print("\n", column)
    print("Mean   :", mean(values))
    print("Median :", median(values))
    print("Mode   :", mode(values))


# --------------------------------------------------
# IMPUTE MISSING VALUES
# --------------------------------------------------

df = impute_data(df, "mean")


# --------------------------------------------------
# CREATE X AND Y
# --------------------------------------------------

X = df[numeric_features].values.tolist()
y = df["Response"].values.tolist()


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

def normalize(X_train, X_test):

    X_train = [row.copy() for row in X_train]
    X_test = [row.copy() for row in X_test]

    columns = len(X_train[0])

    for j in range(columns):

        values = []

        for i in range(len(X_train)):
            values.append(X_train[i][j])

        min_value = min(values)
        max_value = max(values)

        for i in range(len(X_train)):

            if max_value != min_value:
                X_train[i][j] = (
                    X_train[i][j] - min_value
                ) / (max_value - min_value)

        for i in range(len(X_test)):

            if max_value != min_value:
                X_test[i][j] = (
                    X_test[i][j] - min_value
                ) / (max_value - min_value)

    return X_train, X_test


X_train, X_test = normalize(X_train, X_test)


# --------------------------------------------------
# RUN KNN
# --------------------------------------------------

k = 5

predictions = knn(
    X_train,
    y_train,
    X_test,
    k,
    "bubble"
)

print("\nA1 - NORMAL KNN")
print("Actual Class    :", y_test[0])
print("Predicted Class :", predictions[0])


# ==================================================
# A2 : WEIGHTED KNN
# ==================================================

# --------------------------------------------------
# WEIGHTED CLASS ASSIGNMENT
# --------------------------------------------------

def weighted_classify(neighbors):

    class_weights = {}

    for distance, label, index in neighbors:

        # Calculate weight using inverse distance
        if distance == 0:
            weight = float("inf")
        else:
            weight = 1 / distance

        if label not in class_weights:
            class_weights[label] = 0

        class_weights[label] += weight

    # Find class having maximum total weight
    max_weight = max(class_weights.values())

    winners = []

    for label in class_weights:

        if class_weights[label] == max_weight:
            winners.append(label)

    # Tie breaking:
    # choose the class of the closest neighbour
    if len(winners) > 1:

        for distance, label, index in neighbors:

            if label in winners:
                return label

    return winners[0]


# --------------------------------------------------
# WEIGHTED KNN CLASSIFIER
# --------------------------------------------------

def weighted_knn(X_train, y_train, X_test, k=5,
                 sorting_algorithm="bubble"):

    predictions = []

    for test_point in X_test:

        # Reuse A1 neighbour identification module
        neighbors = identify_neighbors(
            X_train,
            y_train,
            test_point,
            k,
            sorting_algorithm
        )

        # Only class assignment is changed
        prediction = weighted_classify(neighbors)

        predictions.append(prediction)

    return predictions


# --------------------------------------------------
# RUN WEIGHTED KNN
# --------------------------------------------------

k = 5

weighted_predictions = weighted_knn(
    X_train,
    y_train,
    X_test,
    k,
    "bubble"
)


# --------------------------------------------------
# DISPLAY NEIGHBOURS AND WEIGHTS
# --------------------------------------------------

# Reuse A1 identify_neighbors() function
neighbors = identify_neighbors(
    X_train,
    y_train,
    X_test[0],
    k,
    "bubble"
)

print("\nA2 - WEIGHTED KNN")
print("\nK NEAREST NEIGHBOURS")

for i, neighbor in enumerate(neighbors):

    distance, label, index = neighbor

    # Inverse distance weighting
    if distance == 0:
        weight = float("inf")
    else:
        weight = 1 / distance

    print(
        "Neighbour", i + 1,
        "Distance =", distance,
        "Class =", label,
        "Weight =", weight
    )


# --------------------------------------------------
# CALCULATE TOTAL WEIGHT OF EACH CLASS
# --------------------------------------------------

class_weights = {}

for distance, label, index in neighbors:

    if distance == 0:
        weight = float("inf")
    else:
        weight = 1 / distance

    if label not in class_weights:
        class_weights[label] = 0

    class_weights[label] += weight


print("\nCLASS WEIGHTS")

for label in class_weights:

    print(
        "Class", label,
        "Total Weight =", class_weights[label]
    )


# --------------------------------------------------
# FINAL WEIGHTED PREDICTION
# --------------------------------------------------

print("\nActual Class    :", y_test[0])
print("Predicted Class :", weighted_predictions[0])

# ==================================================
# A3 : TRAIN AND TEST DATA SPLIT
# ==================================================

import numpy as np
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# CREATE X AND Y
# --------------------------------------------------

X = df[numeric_features].values
y = df["Response"].values


# --------------------------------------------------
# DIVIDE DATA INTO TRAIN AND TEST
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3
)


# Convert NumPy arrays to lists
# so that A1 and A2 functions can reuse them

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()


print("\nA3 - TRAIN TEST SPLIT")

print("Total samples    :", len(X))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

# ==================================================
# A4 : TRAIN KNN CLASSIFIER
# ==================================================

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)

print("\nA4 - KNN CLASSIFIER")
print("K value :", 3)


# ==================================================
# A5 : FIND NEIGH SCORE
# ==================================================

neighscore = neigh.score(X_test, y_test)

print("\nA5 - NEIGH SCORE")
print("Score :", neighscore)
print("Accuracy :", neighscore * 100, "%")


# ==================================================
# A6 : FIND NEIGH PREDICT
# ==================================================

neighpredict = neigh.predict(X_test)

print("\nA6 - NEIGH PREDICT")
print("Predicted Classes :")
print(neighpredict)

# ==================================================
# A7 : KNN WITHOUT USING INBUILT KNN LIBRARY
# ==================================================


# --------------------------------------------------
# FIT FUNCTION
# --------------------------------------------------

def fit(X_train, y_train):

    # Store the training data
    model = {}

    model["X_train"] = X_train
    model["y_train"] = y_train

    return model


# --------------------------------------------------
# PREDICT FUNCTION
# --------------------------------------------------

def predict(model, X_test, k=3,
            sorting_algorithm="bubble"):

    X_train = model["X_train"]
    y_train = model["y_train"]

    predictions = []

    for test_point in X_test:

        # Reuse A1 function to find neighbours
        neighbors = identify_neighbors(
            X_train,
            y_train,
            test_point,
            k,
            sorting_algorithm
        )

        # Reuse A1 majority voting function
        prediction = classify(neighbors)

        predictions.append(prediction)

    return predictions


# --------------------------------------------------
# SCORE FUNCTION
# --------------------------------------------------

def score(model, X_test, y_test, k=3,
          sorting_algorithm="bubble"):

    # Get predictions
    predictions = predict(
        model,
        X_test,
        k,
        sorting_algorithm
    )

    correct = 0

    for i in range(len(y_test)):

        if predictions[i] == y_test[i]:
            correct += 1

    accuracy = correct / len(y_test)

    return accuracy


# ==================================================
# TRAIN THE MODEL USING FIT()
# ==================================================

my_knn = fit(X_train, y_train)


# ==================================================
# PREDICT USING PREDICT()
# ==================================================

mypredict = predict(
    my_knn,
    X_test,
    k=3,
    sorting_algorithm="bubble"
)


print("\nA7 - PREDICT")
print("Predicted Classes:")
print(mypredict)


# ==================================================
# FIND SCORE USING SCORE()
# ==================================================

myscore = score(
    my_knn,
    X_test,
    y_test,
    k=3,
    sorting_algorithm="bubble"
)


print("\nA7 - SCORE")
print("Score :", myscore)
print("Accuracy :", myscore * 100, "%")
# ==================================================
# A8 : COMPARE PACKAGED KNN WITH CREATED KNN
# ==================================================

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# K VALUES TO TEST
# --------------------------------------------------

k_values = range(1, 16)

packaged_accuracy = []
created_accuracy = []


# --------------------------------------------------
# TEST DIFFERENT K VALUES
# --------------------------------------------------

for k in k_values:

    # ----------------------------------------------
    # PACKAGED KNN
    # ----------------------------------------------

    neigh = KNeighborsClassifier(
        n_neighbors=k
    )

    neigh.fit(X_train, y_train)

    score1 = neigh.score(
        X_test,
        y_test
    )

    packaged_accuracy.append(score1)


    # ----------------------------------------------
    # CREATED KNN
    # ----------------------------------------------

    my_knn = fit(
        X_train,
        y_train
    )

    score2 = score(
        my_knn,
        X_test,
        y_test,
        k=k,
        sorting_algorithm="bubble"
    )

    created_accuracy.append(score2)


# --------------------------------------------------
# DISPLAY ACCURACY VALUES
# --------------------------------------------------

print("\nA8 - ACCURACY COMPARISON")

print("\nK\tPackaged KNN\tCreated KNN")

for i in range(len(k_values)):

    print(
        k_values[i],
        "\t",
        packaged_accuracy[i],
        "\t",
        created_accuracy[i]
    )


# --------------------------------------------------
# PLOT ACCURACY VS K
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    packaged_accuracy,
    marker="o",
    label="Packaged KNN"
)

plt.plot(
    k_values,
    created_accuracy,
    marker="s",
    label="Created KNN"
)

plt.xlabel("Value of K")
plt.ylabel("Accuracy")

plt.title("Packaged KNN vs Created KNN")

plt.xticks(list(k_values))

plt.legend()

plt.grid()

plt.show()
# ==================================================
# A9 : WEIGHTED KNN AND COMPARISON WITH A8
# ==================================================

# --------------------------------------------------
# STORE WEIGHTED KNN ACCURACY
# --------------------------------------------------

weighted_accuracy = []


# --------------------------------------------------
# TEST DIFFERENT K VALUES
# --------------------------------------------------

for k in k_values:

    # ----------------------------------------------
    # WEIGHTED KNN
    # Reuse weighted_knn() developed in A2
    # ----------------------------------------------

    weighted_predictions = weighted_knn(
        X_train,
        y_train,
        X_test,
        k=k,
        sorting_algorithm="bubble"
    )


    # ----------------------------------------------
    # CALCULATE ACCURACY
    # ----------------------------------------------

    correct = 0

    for i in range(len(y_test)):

        if weighted_predictions[i] == y_test[i]:
            correct += 1

    accuracy = correct / len(y_test)

    weighted_accuracy.append(accuracy)


# --------------------------------------------------
# DISPLAY A8 AND A9 RESULTS
# --------------------------------------------------

print("\nA9 - WEIGHTED KNN COMPARISON")

print("\nK\tPackaged KNN\tCreated KNN\tWeighted KNN")

for i in range(len(k_values)):

    print(
        k_values[i],
        "\t",
        packaged_accuracy[i],
        "\t",
        created_accuracy[i],
        "\t",
        weighted_accuracy[i]
    )


# --------------------------------------------------
# PLOT A8 AND A9 RESULTS
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    k_values,
    packaged_accuracy,
    marker="o",
    label="Packaged KNN"
)

plt.plot(
    k_values,
    created_accuracy,
    marker="s",
    label="Created KNN"
)

plt.plot(
    k_values,
    weighted_accuracy,
    marker="^",
    label="Weighted KNN"
)

plt.xlabel("Value of K")
plt.ylabel("Accuracy")

plt.title(
    "Comparison of Packaged KNN, Created KNN and Weighted KNN"
)

plt.xticks(list(k_values))

plt.legend()

plt.grid()

plt.show()