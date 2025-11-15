# Homework 1: KNN, Decision Tree, Perceptron from Scratch
# CSCI 4380 Data Mining – Fall 2025
# Author: Gilbert Baraka
# Libraries: Only NumPy, pandas, matplotlib

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------
# TASK 1: KNN on pump_data.csv
# ----------------------------

def load_pump_data():
    df = pd.read_csv('pump_data.csv')
    df = df.drop(columns=['crop'])
    df = df.dropna()
    return df

def train_test_split_manual(X, y, test_size=0.4):
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def knn_predict(X_train, y_train, x_test, k=3, distance='euclidean'):
    distances = []
    for i, x_train in enumerate(X_train):
        if distance == 'euclidean':
            d = euclidean_distance(x_test, x_train)
        else:
            d = manhattan_distance(x_test, x_train)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    vote = Counter(k_nearest).most_common(1)[0][0]
    return vote

def plot_pump_data(X, y, title="Pump Data"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Pump OFF (0)', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Pump ON (1)', alpha=0.7)
    plt.xlabel('Moisture')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

def task1():
    print("=== TASK 1: KNN on pump_data.csv ===")
    df = load_pump_data()
    X = df[['moisture', 'temp']].values.astype(float)
    y = df['pump'].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.4)

    # Plot raw data
    plot_pump_data(X, y, "Raw Pump Data")

    # Evaluate KNN
    for k in [3, 5, 7]:
        for dist in ['euclidean', 'manhattan']:
            y_pred = [knn_predict(X_train, y_train, x, k=k, distance=dist) for x in X_test]
            acc, prec, rec, f1 = evaluate(y_test, y_pred)
            print(f"K={k}, Distance={dist} → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # Plot decision boundary (only for best config, e.g., K=3, euclidean)
    plot_knn_decision_boundary(X_train, y_train, X_test, y_test, k=3, distance='euclidean')
    print()

# ----------------------------
# TASK 2: Decision Tree on Invistico_Airline.csv
# ----------------------------

def load_airline_data():
    df = pd.read_csv('Invistico_Airline.csv', header=None, low_memory=False)
    columns = [
        'satisfaction', 'Customer Type', 'Age', 'Type of Travel', 'Class',
        'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
        'Food and drink', 'Gate location', 'Inflight wifi service',
        'Inflight entertainment', 'Online support', 'Ease of Online booking',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Cleanliness', 'Online boarding',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]
    df.columns = columns
    df = df[['satisfaction', 'Customer Type', 'Type of Travel', 'Class']].copy()
    df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})
    df = df.dropna()
    return df

def entropy(y):
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(y, y_left, y_right):
    H_parent = entropy(y)
    H_children = (len(y_left)/len(y)) * entropy(y_left) + (len(y_right)/len(y)) * entropy(y_right)
    return H_parent - H_children

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, features):
    if len(set(y)) == 1:
        return Node(value=y[0])
    if len(features) == 0:
        majority = Counter(y).most_common(1)[0][0]
        return Node(value=majority)
    
    best_gain = -1
    best_feature = None
    best_threshold = None
    best_left_idx = None
    best_right_idx = None
    
    for feat_idx in features:
        unique_vals = np.unique(X[:, feat_idx])
        for val in unique_vals:
            left_idx = np.where(X[:, feat_idx] == val)[0]
            right_idx = np.where(X[:, feat_idx] != val)[0]
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            gain = information_gain(y, y[left_idx], y[right_idx])
            if gain > best_gain:
                best_gain = gain
                best_feature = feat_idx
                best_threshold = val
                best_left_idx = left_idx
                best_right_idx = right_idx
    
    if best_gain <= 0 or best_feature is None:
        majority = Counter(y).most_common(1)[0][0]
        return Node(value=majority)
    
    remaining_features = [f for f in features if f != best_feature]
    left_subtree = build_tree(X[best_left_idx], y[best_left_idx], remaining_features)
    right_subtree = build_tree(X[best_right_idx], y[best_right_idx], remaining_features)
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

def predict_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] == node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

def print_tree(node, feature_names, depth=0):
    indent = "  " * depth
    if node.value is not None:
        label = "satisfied" if node.value == 1 else "dissatisfied"
        print(f"{indent}→ Predict: {label}")
    else:
        feat_name = feature_names[node.feature]
        print(f"{indent}if {feat_name} == '{node.threshold}':")
        print_tree(node.left, feature_names, depth + 1)
        print(f"{indent}else:  # {feat_name} != '{node.threshold}'")
        print_tree(node.right, feature_names, depth + 1)

def task2():
    print("=== TASK 2: Decision Tree on Invistico_Airline.csv ===")
    df = load_airline_data()
    print("Satisfaction class distribution:")
    print(df['satisfaction'].value_counts(normalize=True))
    
    feature_cols = ['Customer Type', 'Type of Travel', 'Class']
    X_encoded = np.zeros((len(df), len(feature_cols)), dtype=int)
    decoders = {}
    
    for i, col in enumerate(feature_cols):
        unique_vals = df[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        reverse_mapping = {idx: val for val, idx in mapping.items()}
        decoders[i] = reverse_mapping
        X_encoded[:, i] = df[col].map(mapping).values
    
    y = df['satisfaction'].values.astype(int)
    
    # Train-test split
    n = len(X_encoded)
    indices = np.random.permutation(n)
    split = int(n * 0.75)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X_encoded[train_idx], X_encoded[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print("Train label distribution:", np.bincount(y_train))
    print("Test label distribution:", np.bincount(y_test))
    
    root = build_tree(X_train, y_train, features=[0, 1, 2])
    
    def decode_tree(node, decoders):
        if node.value is not None:
            return node
        original_val = decoders[node.feature][node.threshold]
        return Node(
            feature=node.feature,
            threshold=original_val,
            left=decode_tree(node.left, decoders),
            right=decode_tree(node.right, decoders),
            value=None
        )
    
    decoded_root = decode_tree(root, decoders)
    feature_names = ['Customer Type', 'Type of Travel', 'Class']
    print("\nLearned Decision Tree Structure:")
    print_tree(decoded_root, feature_names)
    print()
    
    y_pred = [predict_tree(root, x) for x in X_test]
    acc, prec, rec, f1 = evaluate(y_test, y_pred)
    print(f"Decision Tree → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n")
    

def plot_knn_decision_boundary(X_train, y_train, X_test, y_test, k=3, distance='euclidean'):
    # Create a mesh
    h = 20  # step size
    x_min, x_max = X_train[:, 0].min() - 50, X_train[:, 0].max() + 50
    y_min, y_max = X_train[:, 1].min() - 5, X_train[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict on mesh
    Z = []
    for point in mesh_points:
        pred = knn_predict(X_train, y_train, point, k=k, distance=distance)
        Z.append(pred)
    Z = np.array(Z).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['red', 'green'])
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', marker='o', label='Train OFF')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', marker='o', label='Train ON')
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='red', marker='x', s=80, label='Test OFF')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='green', marker='x', s=80, label='Test ON')
    plt.xlabel('Moisture')
    plt.ylabel('Temperature')
    plt.title(f'KNN Decision Boundary (K={k}, {distance.capitalize()})')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------
# TASK 3: Perceptron
# ----------------------------

def load_perceptron_data():
    train_df = pd.read_csv('perceptron-train.csv')
    test_df = pd.read_csv('perceptron-test.csv')
    train_df.columns = ['Output', 'x1', 'x2']
    test_df.columns = ['Output', 'x1', 'x2']
    return train_df.dropna(), test_df.dropna()

def perceptron_train(X, y, lr=0.01, epochs=100):
    n_samples, n_features = X.shape
    X = np.c_[np.ones(n_samples), X]
    w = np.random.uniform(-0.01, 0.01, size=n_features + 1)
    y = np.where(y == -1, -1, 1)  # ensure -1/1
    
    for _ in range(epochs):
        for i in range(n_samples):
            linear = np.dot(w, X[i])
            pred = 1 if linear >= 0 else -1
            if pred != y[i]:
                w += lr * y[i] * X[i]
    return w

def perceptron_predict(w, X):
    X = np.c_[np.ones(X.shape[0]), X]
    preds = []
    for x in X:
        linear = np.dot(w, x)
        pred = 1 if linear >= 0 else -1
        preds.append(pred)
    return np.array(preds)

def task3():
    print("=== TASK 3: Perceptron ===")
    train_df, test_df = load_perceptron_data()
    print("Train label distribution:")
    print(train_df['Output'].value_counts(normalize=True))
    print("Test label distribution:")
    print(test_df['Output'].value_counts(normalize=True))
    
    X_train = train_df[['x1', 'x2']].values
    y_train = train_df['Output'].values
    X_test = test_df[['x1', 'x2']].values
    y_test = test_df['Output'].values
    
    for lr in [0.005, 0.01, 0.05]:
        w = perceptron_train(X_train, y_train, lr=lr, epochs=100)
        y_pred = perceptron_predict(w, X_test)
        acc, prec, rec, f1 = evaluate(y_test, y_pred)
        print(f"LR={lr}, Act=sign → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        
        # Plot inside the loop, right after training
        plot_perceptron_boundary(w, X_train, y_train, X_test, y_test, 
                                title=f"Perceptron Decision Boundary (LR={lr})")
    print()

def plot_perceptron_boundary(w, X_train, y_train, X_test, y_test, title="Perceptron Decision Boundary"):
    plt.figure(figsize=(10, 8))
    
    # Plot data
    y_train_plot = np.where(y_train == -1, 0, 1)
    y_test_plot = np.where(y_test == -1, 0, 1)
    plt.scatter(X_train[y_train_plot == 0, 0], X_train[y_train_plot == 0, 1], c='red', marker='o', label='Train -1')
    plt.scatter(X_train[y_train_plot == 1, 0], X_train[y_train_plot == 1, 1], c='green', marker='o', label='Train 1')
    plt.scatter(X_test[y_test_plot == 0, 0], X_test[y_test_plot == 0, 1], c='red', marker='x', s=80, label='Test -1')
    plt.scatter(X_test[y_test_plot == 1, 0], X_test[y_test_plot == 1, 1], c='green', marker='x', s=80, label='Test 1')
    
    # Plot decision boundary: w0 + w1*x1 + w2*x2 = 0 → x2 = -(w0 + w1*x1)/w2
    x_vals = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
    if abs(w[2]) > 1e-6:  # avoid division by zero
        y_vals = -(w[0] + w[1] * x_vals) / w[2]
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    print("Running Homework 1...\n")
    task1()
    task2()
    task3()
    print("All tasks completed.")