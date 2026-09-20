import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


df = pd.read_excel(r"C:\ISHU\Education\Amrita\5th_Sem\ML\Assignment\Material\thyroid_dataset.xlsx")

cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[cols] = df[cols].fillna(df[cols].median())

df["Condition"] = df["Condition"].map({
    "NO CONDITION": 0,
    "CONDITION": 1
})

X = df[cols].values
y = df["Condition"].values


# A1
def bin_data(data, bins=4, method="equal"):
    data = np.asarray(data)

    if method == "equal":
        edges = np.linspace(data.min(), data.max(), bins + 1)
        result = np.digitize(data, edges[1:-1])

    elif method == "frequency":
        edges = np.quantile(data, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        result = np.digitize(data, edges[1:-1])

    else:
        raise ValueError("Method must be 'equal' or 'frequency'")

    return result


def ent(y):
    values, counts = np.unique(y, return_counts=True)
    p = counts / len(y)

    return -np.sum(p * np.log2(p))


# A2
def gini(y):
    values, counts = np.unique(y, return_counts=True)
    p = counts / len(y)

    return 1 - np.sum(p ** 2)


# A3
def info_gain(feature, target):
    total = ent(target)
    weighted = 0

    for value in np.unique(feature):
        part = target[feature == value]

        weight = len(part) / len(target)
        weighted += weight * ent(part)

    return total - weighted


def root_feature(X, y, names):
    gains = []

    for i in range(X.shape[1]):
        gain = info_gain(X[:, i], y)
        gains.append(gain)

    best = np.argmax(gains)

    return names[best], gains


# A4
def make_bins(X, bins=4, method="equal"):
    new_X = np.zeros_like(X, dtype=int)

    for i in range(X.shape[1]):
        new_X[:, i] = bin_data(
            X[:, i],
            bins,
            method
        )

    return new_X


# A5
def make_tree(X_train, y_train, criterion="entropy"):
    tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=4,
        random_state=42
    )

    tree.fit(X_train, y_train)

    return tree


# A6
def show_tree(tree, names):
    plt.figure(figsize=(16, 9))

    plot_tree(
        tree,
        feature_names=names,
        class_names=["NO CONDITION", "CONDITION"],
        filled=True,
        rounded=True
    )

    plt.title("Decision Tree")
    plt.show()

# A7
def boundary(X, y, names):
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42
    )

    tree.fit(X, y)

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 100),
        np.linspace(x2_min, x2_max, 100)
    )

    pred = tree.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    pred = pred.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        pred,
        alpha=0.3
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolor="k"
    )

    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title("Decision Tree Decision Boundary")

    plt.show()
    
# A8
def tune_tree(X_train, y_train):
    tree = DecisionTreeClassifier(random_state=42)

    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        tree,
        params,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    return grid


if __name__ == "__main__":

    print("Dataset Shape:", X.shape)

    # A1
    print("\nA1")

    print("Entropy =", ent(y))

    binned = make_bins(
        X,
        bins=4,
        method="equal"
    )

    print("Binned Data:")
    print(binned[:5])

    # A2
    print("\nA2")

    print("Gini Index =", gini(y))

    # A3
    print("\nA3")

    root, gains = root_feature(
        binned,
        y,
        cols
    )

    for name, gain in zip(cols, gains):
        print(name, "=", gain)

    print("Root Feature =", root)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # A5
    print("\nA5")

    tree = make_tree(
        X_train,
        y_train,
        criterion="entropy"
    )

    pred = tree.predict(X_test)

    print("Accuracy =", accuracy_score(y_test, pred))

    # A6
    print("\nA6")

    show_tree(
        tree,
        cols
    )

    # A7
    print("\nA7")

    two = X[:, [0, 1]]

    boundary(
        two,
        y,
        [cols[0], cols[1]]
    )

    # A8
    print("\nA8")

    grid = tune_tree(
        X_train,
        y_train
    )

    print("Best Parameters:")
    print(grid.best_params_)

    print("Best Cross Validation Score:")
    print(grid.best_score_)

    best = grid.best_estimator_

    test_pred = best.predict(X_test)

    print("Tuned Model Test Accuracy:")
    print(accuracy_score(y_test, test_pred))