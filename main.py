import numpy as np
import pandas as pd
from sklearn.neural_network import  MLPClassifier


# Generate train (and test) data. There are two input variable (A, B). Target variable is A XOR B
data = pd.DataFrame([(a, b, a ^ b) for a, b in zip(np.random.randint(2, size=20), np.random.randint(2, size=20))], columns=["A", "B", "X"])

# Separate features and target
X, y = data[["A", "B"]], data["X"]

# Initialize and train a classification model
c = MLPClassifier(max_iter=1000).fit(X, y)

# Calculate mean accuracy using train data (there were no train-test split)
print(f"Accuracy score: {c.score(X, y)}")

# Get model prediction for different inputs
for a in [0, 1]:
    for b in [0, 1]:
        print("-" * 50)
        print(f"Inputs: {a}, {b}")
        print(f"True output: {a ^ b}")
        print(f"Model output: {c.predict([[a, b]])}")
