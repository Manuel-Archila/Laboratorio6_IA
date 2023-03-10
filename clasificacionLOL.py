import pandas as pd
import numpy as np
from arbol import Arbol
from metricas import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def clasificacion(path):

    df = pd.read_csv(path)

    
    # Se ve si el dataset esta balanceado

    print(df["blueWins"].value_counts())

    # Se ve que variables estan relacionadas con status

    columnas = []
    for column in df.columns:
        if column != "blueWins":
            try:
                correlation = df[column].corr(df["blueWins"])
                if correlation > 0.1 or correlation < -0.1:
                    columnas.append(column)
            except:
                pass
        else:
            columnas.append(column)


    # Se crea el dataset con las columnas seleccionadas

    df = df[columnas]

    # Normalizar las columnas

    for column in df.columns:
        if column != "blueWins":
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    # Se separa el dataset en training y testing
    copy = df

    X = copy.drop("blueWins", axis=1)
    y = df["blueWins"]
    

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    clf = Arbol(max_depth=2)
    clf.fit(X_train, np.array(y_train))


    prediction = clf.predict(np.array(X_test))


    # mejores = clf.top5()

    print ("Mejores: ", features)

    print("Accuracy: ", metrica(prediction , y_test))

    # Se encuentran los 5 elementos arriba en el arbol
    arbol = clf.tree_


    # Se vuelve a crear el modelo utilizando la libreria de sklearn

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    libreria = DecisionTreeClassifier(max_depth=2)
    libreria.fit(X_train, y_train)

    prediction = libreria.predict(X_test)

    print("Accuracy con libreria: ", accuracy_score(y_test, prediction))


    
    pass