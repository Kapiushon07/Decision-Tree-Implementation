import pandas as pd
import numpy as np
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
np.random.seed(42)

def train_n_test(X, y, max_depth=17):
    s = int(0.7 * len(X))  # split
    mymodel = DecisionTree(criterion="information_gain", max_depth=max_depth)

    # Training
    X_train = pd.DataFrame(X[0:s]).reset_index(drop=True)
    y_train = pd.Series(y[0:s], dtype="category").reset_index(drop=True)
    mymodel.fit(X_train, y_train)

    # Test
    X_test = pd.DataFrame(X[s:]).reset_index(drop=True)
    y_test = pd.Series(y[s:]).reset_index(drop=True)

    y_pred = mymodel.predict(X_test)

    print("Accuracy:", accuracy(pd.Series(y_pred), y_test))

    for cls in y_test.unique():
        print(f'Precision -- class {cls} =', precision(y_pred, y_test, cls))
        print(f'Recall -- class {cls} =:', recall(y_pred, y_test, cls))


# ---------- Part (b) ----------
def five_fold_validation(X, y, depth=5):
    """
    Perform 5-fold cross validation with a decision tree.
    Works for any dataset size.
    """
    k = 5
    n = len(X)
    fold_size = n // k
    accuracies = []

    for fold in range(k):
        # Determine test indices for this fold
        start, end = fold * fold_size, (fold + 1) * fold_size
        test_idx = list(range(start, end))
        train_idx = [i for i in range(n) if i not in test_idx]

        # Split into training and test sets
        X_train, y_train = pd.DataFrame(X[train_idx]), pd.Series(y[train_idx], dtype="category")
        X_test, y_test = pd.DataFrame(X[test_idx]), pd.Series(y[test_idx])

        # Train decision tree
        model = DecisionTree(criterion="information_gain", max_depth=depth)
        model.fit(X_train, y_train)

        # Predictions and accuracy
        y_pred = model.predict(X_test)
        acc = accuracy(pd.Series(y_pred), y_test)
        accuracies.append(acc)

        print(f"Fold {fold+1}: accuracy = {acc:.3f}")

    # Final average across folds
    print("Cross-validation results:")
    print("Accuracies per fold:", [round(a, 3) for a in accuracies])
    print("Mean accuracy:", round(np.mean(accuracies), 3))


def nested_validation(X, y, depth_range=range(1, 11), outer_folds=5):
    """
    Nested cross-validation:
    - Outer CV splits dataset into train/test.
    - Inner CV chooses the best depth from depth_range.
    """
    n = len(X)
    fold_size = n // outer_folds

    for outer in range(outer_folds):
        # Outer split
        test_idx = list(range(outer * fold_size, (outer + 1) * fold_size))
        train_idx = [i for i in range(n) if i not in test_idx]

        X_train = pd.DataFrame(X[train_idx]).reset_index(drop=True)
        y_train = pd.Series(y[train_idx], dtype="category").reset_index(drop=True)
        X_test = pd.DataFrame(X[test_idx]).reset_index(drop=True)
        y_test = pd.Series(y[test_idx]).reset_index(drop=True)

        # Inner CV for hyperparameter tuning
        inner_folds = outer_folds - 1
        inner_fold_size = len(X_train) // inner_folds
        mean_acc_per_depth = {}

        for depth in depth_range:
            inner_accs = []
            for inner in range(inner_folds):
                val_idx = list(range(inner * inner_fold_size, (inner + 1) * inner_fold_size))
                tr_idx = [i for i in range(len(X_train)) if i not in val_idx]

                X_inner = pd.DataFrame(X_train.iloc[tr_idx]).reset_index(drop=True)
                y_inner = y_train.iloc[tr_idx].reset_index(drop=True)
                X_val = pd.DataFrame(X_train.iloc[val_idx]).reset_index(drop=True)
                y_val = y_train.iloc[val_idx].reset_index(drop=True)

                model = DecisionTree(criterion="information_gain", max_depth=depth)
                model.fit(X_inner, y_inner)
                y_val_pred = model.predict(X_val)

                inner_accs.append(accuracy(pd.Series(y_val_pred), y_val))

            mean_acc_per_depth[depth] = np.mean(inner_accs)

        # Select best depth
        best_depth = max(mean_acc_per_depth, key=mean_acc_per_depth.get)

        # Retrain with best depth on all outer training data
        final_model = DecisionTree(criterion="information_gain", max_depth=best_depth)
        final_model.fit(X_train, y_train)
        y_test_pred = final_model.predict(X_test)

        print(f"Outer fold {outer+1}: accuracy = {accuracy(pd.Series(y_test_pred), y_test):.3f}, best depth = {best_depth}")


# main ---------
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

print("\nQuestion 2) a)\n")
train_n_test(X, y)
print("\nEnd of Question 2) a)\n")


print("\nb - Five Fold Cross Validation\n")
five_fold_validation(X, y)
print("\nend of Five Fold Cross Validation\n")

print("\nb-Nested Cross Validation\n")
nested_validation(X, y)
print("\nEnd of Nested Cross Validation\n")