import numpy as np
from collections import Counter

# DecisionTree class
class DecisionTree:
    def __init__(self, max_depth, min_samples_split, criterion, max_features):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(x, self.tree) for x in X]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return leaf_value

        feat_idxs = np.random.choice(n_features, self.max_features, replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return leaf_value

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {"feature_index": best_feat, "threshold": best_thresh, "left": left, "right": right}

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold) if self.criterion == 'information_gain' else self._gini_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def _gini_gain(self, y, X_column, split_thresh):
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        gini_impurity = (n_l / n) * g_l + (n_r / n) * g_r
        return gini_impurity

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _gini(self, y):
        hist = np.bincount(y)
        n = len(y)
        return 1 - np.sum([(i / n)**2 for i in hist])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _predict(self, x, tree):
        if isinstance(tree, dict):
            feature_val = x[tree["feature_index"]]
            if feature_val <= tree["threshold"]:
                return self._predict(x, tree["left"])
            else:
                return self._predict(x, tree["right"])
        else:
            return tree

# RandomForest class    
class RandomForest:
    def __init__(self, n_trees, max_depth, min_samples_split, criterion, max_features, bootstrap=True, oob_score=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.trees = []
        self.oob_score_ = None

    def fit(self, X, y):
        self.trees = []
        self.n_features = X.shape[1]

        if self.bootstrap:
            for _ in range(self.n_trees):
                tree = DecisionTree(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    criterion=self.criterion,
                                    max_features=self.max_features)
                X_sample, y_sample = self._bootstrap_samples(X, y)
                tree.fit(X_sample, y_sample)
                self.trees.append(tree)
        else:
            for _ in range(self.n_trees):
                tree = DecisionTree(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    criterion=self.criterion,
                                    max_features=self.max_features)
                tree.fit(X, y)
                self.trees.append(tree)

        if self.oob_score:
            self._calculate_oob_score(X, y)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._most_common_label(pred) for pred in predictions.T])

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        if y is None:
            return idxs
        else:
            return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _calculate_oob_score(self, X, y):
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, self.n_trees))
        for i, tree in enumerate(self.trees):
            idxs = self._bootstrap_samples(np.arange(n_samples), None)
            mask = np.ones(n_samples, dtype=bool)
            mask[idxs] = False
            oob_predictions[mask, i] = tree.predict(X[mask])

        oob_predictions_majority = np.array([self._most_common_label(pred) for pred in oob_predictions])
        self.oob_score_ = np.mean(oob_predictions_majority == y)

    def feature_importances(self):
        importances = np.zeros(self.n_features)
        for tree in self.trees:
            importances += self._feature_importances(tree.tree)
        importances /= self.n_trees
        return importances

    def _feature_importances(self, tree):
        if isinstance(tree, dict):
            importances = np.zeros(self.n_features)
            importances[tree["feature_index"]] = 1
            left_importances = self._feature_importances(tree["left"])
            right_importances = self._feature_importances(tree["right"])
            importances += left_importances + right_importances
            return importances
        else:
            return np.zeros(self.n_features)


 ##################
 # Implementation #
 ##################
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target 

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a random forest classifier instance
    rf = RandomForest(n_trees=150, max_depth=30, min_samples_split=5, criterion='gini', max_features=4, bootstrap=True, oob_score=True)

    # Train the random forest classifier
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print the out-of-bag score
    if rf.oob_score:
        print(f"Out-of-bag score: {rf.oob_score_:.2f}")

    # Print the feature importances
    importances = rf.feature_importances()
    print("Feature Importances:")
    for i, importance in enumerate(importances):
        print(f"Feature {i+1}: {importance:.2f}")