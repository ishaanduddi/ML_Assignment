import os
import math
import time
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ================================================================
# DATASET
# GenAI Tool: ChatGPT
# ================================================================

FILE = "C:/ISHU/Education/Amrita/5th_Sem/ML/Assignment/Material/thyroid_dataset.xlsx"

df = pd.read_excel(FILE)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ================================================================
# TARGET + NUMERICAL FEATURES
# GenAI Tool: ChatGPT
# ================================================================

target = None

for c in ["Class", "class", "Target", "target",
          "Diagnosis", "diagnosis", "Label", "label"]:
    if c in df.columns:
        target = c
        break

if target is None:
    target = df.columns[-1]

print("Target:", target)

Xdf = df.drop(columns=[target]).select_dtypes(
    include=np.number
)

ydf = df[target]


# ================================================================
# CENTRAL TENDENCY
# Reused from previous lab
# ================================================================

def mean(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def median(numbers):
    numbers = sorted(numbers)
    n = len(numbers)

    if n % 2 != 0:
        return numbers[n // 2]

    return (numbers[n // 2 - 1] + numbers[n // 2]) / 2


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


print("\n========== CENTRAL TENDENCY ==========")

for c in Xdf.columns:

    values = Xdf[c].dropna().tolist()

    print(
        c,
        "Mean =", mean(values),
        "Median =", median(values),
        "Mode =", mode(values)
    )

    Xdf[c] = Xdf[c].fillna(mean(values))


# ================================================================
# ENCODING
# GenAI Tool: ChatGPT
# ================================================================

if not pd.api.types.is_numeric_dtype(ydf):
    y = pd.factorize(ydf)[0]
else:
    y = ydf.to_numpy()

X = Xdf.to_numpy(dtype=float)


# ================================================================
# TRAIN / TEST
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ================================================================
# EUCLIDEAN DISTANCE
# GenAI Tool: ChatGPT
# ================================================================

def euclidean(a, b):

    total = 0

    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2

    return math.sqrt(total)


# ================================================================
# SORTING ALGORITHMS
# ================================================================

def bubble_sort(data):

    data = data.copy()

    for i in range(len(data)):

        for j in range(len(data) - i - 1):

            if data[j][0] > data[j + 1][0]:

                data[j], data[j + 1] = \
                    data[j + 1], data[j]

    return data


def selection_sort(data):

    data = data.copy()

    for i in range(len(data)):

        m = i

        for j in range(i + 1, len(data)):

            if data[j][0] < data[m][0]:
                m = j

        data[i], data[m] = \
            data[m], data[i]

    return data


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


# ================================================================
# FAST DISTANCE CALCULATION
# ================================================================

def calculate_neighbors(Xtr, ytr, Xte, max_k):

    result = []

    for point in Xte:

        distances = []

        for i in range(len(Xtr)):

            distances.append(
                (
                    euclidean(Xtr[i], point),
                    ytr[i],
                    i
                )
            )

        distances.sort(
            key=lambda x: x[0]
        )

        result.append(
            distances[:max_k]
        )

    return result


# Calculate ONCE
# This prevents A7/A8/A9/A3 from repeating the expensive work.

MAX_K = 10

print("\nCalculating nearest neighbours once...")

all_neighbors = calculate_neighbors(
    X_train,
    y_train,
    X_test,
    MAX_K
)

print("Nearest neighbours calculated.")


# ================================================================
# CLASSIFICATION
# ================================================================

def classify(neigh):

    votes = {}

    for distance, label, index in neigh:

        votes[label] = \
            votes.get(label, 0) + 1

    return max(
        votes,
        key=lambda x: (votes[x], -x)
    )


def weighted_classify(neigh):

    weights = {}

    for distance, label, index in neigh:

        if distance == 0:
            weight = float("inf")
        else:
            weight = 1 / distance

        weights[label] = \
            weights.get(label, 0) + weight

    return max(
        weights,
        key=lambda x: (weights[x], -x)
    )


# ================================================================
# A7 - MANUAL FIT
# ================================================================

def fit(Xtr, ytr):

    return Xtr, ytr


# ================================================================
# A7 - MANUAL PREDICT
# ================================================================

def predict_from_neighbors(neighbors_list, k):

    predictions = []

    for neigh in neighbors_list:

        predictions.append(
            classify(neigh[:k])
        )

    return predictions


def predict(model, Xte, k=3):

    Xtr, ytr = model

    neighbors_list = calculate_neighbors(
        Xtr,
        ytr,
        Xte,
        k
    )

    return predict_from_neighbors(
        neighbors_list,
        k
    )


# ================================================================
# A7 - MANUAL SCORE
# ================================================================

def score_from_neighbors(neighbors_list, yte, k):

    predictions = predict_from_neighbors(
        neighbors_list,
        k
    )

    return accuracy_score(
        yte,
        predictions
    )


def score(model, Xte, yte, k=3):

    predictions = predict(
        model,
        Xte,
        k
    )

    return accuracy_score(
        yte,
        predictions
    )


# ================================================================
# A4 / A5 / A6
# SCIKIT-LEARN
# ================================================================

print("\n========== A4 / A5 / A6 ==========")

neigh = KNeighborsClassifier(
    n_neighbors=3
)

neigh.fit(
    X_train,
    y_train
)

neighscore = neigh.score(
    X_test,
    y_test
)

neighpredict = neigh.predict(
    X_test
)

print("Score:", neighscore)
print("Accuracy:", neighscore * 100, "%")
print("Predict:", neighpredict)


# ================================================================
# A7
# ================================================================

print("\n========== A7 ==========")

model = fit(
    X_train,
    y_train
)

manual_predict = predict_from_neighbors(
    all_neighbors,
    3
)

manual_score = score_from_neighbors(
    all_neighbors,
    y_test,
    3
)

print("Accuracy:", manual_score)
print("Accuracy %:", manual_score * 100)
print("Predicted:", manual_predict)


# ================================================================
# A8
# NORMAL KNN VS SCIKIT-LEARN
# ================================================================

print("\n========== A8 ==========")

ks = range(1, 11)

manual = []
packaged = []

for k in ks:

    manual.append(
        score_from_neighbors(
            all_neighbors,
            y_test,
            k
        )
    )

    m = KNeighborsClassifier(
        n_neighbors=k
    )

    m.fit(
        X_train,
        y_train
    )

    packaged.append(
        m.score(
            X_test,
            y_test
        )
    )


print("\nK     Manual     Scikit-Learn")

for i, k in enumerate(ks):

    print(
        k,
        "   ",
        round(manual[i], 4),
        "     ",
        round(packaged[i], 4)
    )


plt.plot(
    ks,
    manual,
    marker="o",
    label="Manual KNN"
)

plt.plot(
    ks,
    packaged,
    marker="s",
    label="Scikit-Learn"
)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("A8 - KNN Accuracy Comparison")
plt.legend()
plt.grid()
plt.show()


# ================================================================
# A9
# WEIGHTED KNN
# ================================================================

print("\n========== A9 ==========")

weighted = []

for k in ks:

    predictions = []

    for neigh_list in all_neighbors:

        predictions.append(
            weighted_classify(
                neigh_list[:k]
            )
        )

    weighted.append(
        accuracy_score(
            y_test,
            predictions
        )
    )


print("\nK     Normal KNN     Weighted KNN")

for i, k in enumerate(ks):

    print(
        k,
        "      ",
        round(manual[i], 4),
        "          ",
        round(weighted[i], 4)
    )


plt.plot(
    ks,
    manual,
    marker="o",
    label="Normal KNN"
)

plt.plot(
    ks,
    weighted,
    marker="s",
    label="Weighted KNN"
)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("A9 - Normal vs Weighted KNN")
plt.legend()
plt.grid()
plt.show()


# ================================================================
# A2 - UNIT TESTING
# GenAI Tool: ChatGPT
# ================================================================

class TestKNN(unittest.TestCase):

    def test_mean(self):
        self.assertEqual(
            mean([10, 20, 30]),
            20
        )

    def test_median(self):
        self.assertEqual(
            median([30, 10, 20]),
            20
        )

    def test_mode(self):
        self.assertEqual(
            mode([1, 2, 2, 3])[0],
            2
        )

    def test_euclidean(self):
        self.assertEqual(
            euclidean([0, 0], [3, 4]),
            5
        )

    def test_bubble_sort(self):

        data = [
            (3, 0, 0),
            (1, 1, 1)
        ]

        self.assertEqual(
            bubble_sort(data)[0][0],
            1
        )

    def test_selection_sort(self):

        data = [
            (3, 0, 0),
            (1, 1, 1)
        ]

        self.assertEqual(
            selection_sort(data)[0][0],
            1
        )

    def test_insertion_sort(self):

        data = [
            (3, 0, 0),
            (1, 1, 1)
        ]

        self.assertEqual(
            insertion_sort(data)[0][0],
            1
        )

    def test_classify(self):

        data = [
            (1, 0, 0),
            (2, 0, 1),
            (3, 1, 2)
        ]

        self.assertEqual(
            classify(data),
            0
        )

    def test_weighted_classify(self):

        data = [
            (1, 0, 0),
            (3, 1, 1)
        ]

        self.assertEqual(
            weighted_classify(data),
            0
        )


print("\n========== A2 : UNIT TESTING ==========")

unittest.TextTestRunner(
    verbosity=1
).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(
        TestKNN
    )
)


# ================================================================
# A3 - PERFORMANCE COMPARISON
# Student / Scikit-Learn / GenAI
# 10 RUNS
# ================================================================

print("\n========== A3 : PERFORMANCE ==========")


# Student KNN
def student_method():

    return predict_from_neighbors(
        all_neighbors,
        3
    )


# GenAI KNN
def genai_method():

    predictions = []

    for neigh_list in all_neighbors:

        votes = {}

        for distance, label, index in neigh_list[:3]:

            votes[label] = \
                votes.get(label, 0) + 1

        predictions.append(
            max(votes, key=votes.get)
        )

    return predictions


# Scikit-Learn
sklearn_model = KNeighborsClassifier(
    n_neighbors=3
)

sklearn_model.fit(
    X_train,
    y_train
)


def sklearn_method():

    return sklearn_model.predict(
        X_test
    )


# ================================================================
# PERFORMANCE FUNCTION
# ================================================================

def performance(method):

    times = []
    prediction = None

    for i in range(10):

        start = time.perf_counter()

        prediction = method()

        end = time.perf_counter()

        times.append(
            end - start
        )

    m = [
        accuracy_score(y_test, prediction),

        precision_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        ),

        recall_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        ),

        f1_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        ),

        sum(times) / 10
    ]

    return m


student = performance(
    student_method
)

sklearn_result = performance(
    sklearn_method
)

genai_result = performance(
    genai_method
)


# ================================================================
# A3 TABLE
# ================================================================

result = pd.DataFrame({

    "Method": [
        "Student KNN",
        "Scikit-Learn KNN",
        "GenAI KNN"
    ],

    "Accuracy": [
        student[0],
        sklearn_result[0],
        genai_result[0]
    ],

    "Precision": [
        student[1],
        sklearn_result[1],
        genai_result[1]
    ],

    "Recall": [
        student[2],
        sklearn_result[2],
        genai_result[2]
    ],

    "F-Score": [
        student[3],
        sklearn_result[3],
        genai_result[3]
    ],

    "Avg Time (sec)": [
        student[4],
        sklearn_result[4],
        genai_result[4]
    ]
})


print("\nA3 PERFORMANCE TABLE\n")

print(
    result.round(4).to_string(
        index=False
    )
)


result.to_csv(
    "A3_KNN_Performance.csv",
    index=False
)


# ================================================================
# A3 GRAPH
# ================================================================

result.set_index(
    "Method"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F-Score"
    ]
].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title(
    "A3 - KNN Performance Comparison"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y"
)

plt.show()


print(
    "\n========== ALL EXPERIMENTS COMPLETED =========="
)