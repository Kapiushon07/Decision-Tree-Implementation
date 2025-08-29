import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
np.random.seed(42)

cases = {
    "RIRO": {
        "X": lambda N, P: pd.DataFrame(np.random.randn(N, P)),
        "y": lambda N, P: pd.Series(np.random.randn(N))
    },
    "RIDO": {
        "X": lambda N, P: pd.DataFrame(np.random.randn(N, P)),
        "y": lambda N, P: pd.Series(np.random.randint(P, size=N), dtype="category")
    },
    "DIRO": {
        "X": lambda N, P: pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)}),
        "y": lambda N, P: pd.Series(np.random.randn(N))
    },
    "DIDO": {
        "X": lambda N, P: pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)}),
        "y": lambda N, P: pd.Series(np.random.randint(P, size=N), dtype="category")
    }
}

# creating the subplots (4 rows, 2 columns: Fit time n Predict time)
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
fig.suptitle("Decision Tree Fit & Predict Time for Different Data Types", fontsize=16)

for i, (case, funcs) in enumerate(cases.items()):
    learning_time, predict_time = [], []

    for N in range(1, 7):
        for P in range(6, 42):
            X, y = funcs["X"](N, P), funcs["y"](N, P)

            # Fit timing
            start = time.perf_counter()
            tree = DecisionTree(criterion="information_gain")
            tree.fit(X, y)
            learning_time.append(time.perf_counter() - start)

            # Predict timing
            start = time.perf_counter()
            tree.predict(X)
            predict_time.append(time.perf_counter() - start)

    # Plotting respective subplot
    axes[i, 0].plot(learning_time, color='blue')
    axes[i, 0].set_title(f"{case} - Fit Time")
    axes[i, 0].set_ylabel("Time (s)")

    axes[i, 1].plot(predict_time, color='green')
    axes[i, 1].set_title(f"{case} - Predict Time")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()




