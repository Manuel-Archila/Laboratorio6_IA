import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


class Arbol():
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):

        self.columnas = X.columns
        X = np.array(X)
        self.classes = len(set(y)) 
        self.features = X.shape[1]
        self.tree_ = self.crear_arbol(X, y)


    def predict(self, X):
        return [self.predict_input(inputs) for inputs in X]

    def gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.classes))

    def best_split(self, X, y):
        
        m = y.size
        if m <= 1:
            return None, None

        # Conteo de las clases el nodo actual
        num_parent = [np.sum(y == c) for c in range(self.classes)]

        # Se encuentra el Gini del nodo actual
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Se iteran los features
        for idx in range(self.features):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.classes
            num_right = num_parent.copy()
            for i in range(1, m):  # posibles soluciones
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.classes)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  

        return best_idx, best_thr

    def crear_arbol(self, X, y, depth=0):

        num_samples_per_class = [np.sum(y == i) for i in range(self.classes)]

        
        idx, thr = self.best_split(X, y)

        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self.gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class
        )
        node.feature_name = self.columnas[idx]
        

        # Se hace split recursivamente hasta que se alcanza la profundidad maxima.
        if depth < self.max_depth:
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.feature_name = self.columnas[idx]
                node.left = self.crear_arbol(X_left, y_left, depth + 1)
                node.right = self.crear_arbol(X_right, y_right, depth + 1)
        return node

    def predict_input(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def feature_importances(self, X_train, y_train):

        features = X_train.select_dtypes(include=[np.number]).columns

        importance = {}
        for i in range(len(features)):
            # Obtener los índices de las muestras donde la característica i es igual a 1
            idx = np.where(X_train[features[i]] > 0)[0]
            
            # Obtener el impurity del nodo padre
            parent_impurity = self.gini(y_train)
            
            # Calcular el impurity promedio ponderado de los nodos hijos
            child_impurity = 0
            for j in range(2):
                child_idx = np.intersect1d(idx, np.where(y_train == j)[0])
                
                child_weight = len(child_idx) / len(idx)
                child_impurity += child_weight * self.gini(y_train[child_idx])
            
            importance[features[i]] = parent_impurity - child_impurity

        importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
        return importance
    
    # def get_features(self, tree):
    #     visited_nodes = []
    #     names = []
    #     queue = [tree]
    #     while queue:
    #         current_node = queue.pop(0)

    #         dict = {
    #             "name": current_node.feature_name,
    #             "importance": self.feature_importances(current_node),
    #             "predicted_class": current_node.predicted_class,
    #         }

    #         visited_nodes.append(dict)

    #         if current_node.left:
    #             queue.append(current_node.left)
    #         if current_node.right:
    #             queue.append(current_node.right)

            
    #         names.append(current_node.feature_name)

    #     return visited_nodes
    
    # def get_first_five_nodes(self, tree):
    #     visited_nodes = []
    #     names = []
    #     queue = [tree]
    #     while queue and len(visited_nodes) < 5:
    #         current_node = queue.pop(0)

    #         dict = {
    #             "name": current_node.feature_name,
    #             "importance": self.f,
    #             "predicted_class": current_node.predicted_class,
    #         }

    #         # if current_node.feature_name not in names:
    #         visited_nodes.append(dict)

    #         if current_node.left:
    #             queue.append(current_node.left)
    #         if current_node.right:
    #             queue.append(current_node.right)

            
    #         names.append(current_node.feature_name)

    #     return visited_nodes

    # def top5(self):
    #     res = self.get_features(self.tree_)
    #     res = sorted(res, key=lambda x: x["importance"], reverse=True)
    #     res = res[:5]
    #     return res

class Node:
     def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_name = None
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


    