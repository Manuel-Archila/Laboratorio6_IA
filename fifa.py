import pandas as pd
from metricas import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor



def clasificacionFIFA(path):

    df = pd.read_csv(path)

    # Eliminar la columna que contiene el id

    df = df.drop(df.columns[0], axis=1)

    # Se eliminan los valores nulos
    
    df = df.dropna()
    df = df.drop_duplicates()

    # Se ve que variables estan relacionadas con Potential

    columnas = []
    for column in df.columns:
        if column != "Potential":
            try:
                correlation = df[column].corr(df["Potential"])
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
        if column != "Potential":
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    # Se separa el dataset en training y testing
    copy = df

    X = copy.drop("Potential", axis=1)
    y = df["Potential"]
    

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    
    regr = DecisionTreeRegressor(random_state=42)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    importancias = regr.feature_importances_

    importancia = {}
    for i in range(len(importancias)):
        importancia[regr.feature_names_in_[i]] = importancias[i]

    indices_ordenados = sorted(importancia, key=importancia.get, reverse=True)

    print("Las top 5 caracteristicas:")

    for i in range(5):
        print('\t',indices_ordenados[i],' = ', importancia[indices_ordenados[i]])

    

    error = mean_squared_error(y_test, y_pred)

    print("MSE con libreria: ",error)

    # Random forest

    print("\nRandom forest")

    regr = RandomForestClassifier(random_state=42)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    error = mean_squared_error(y_test, y_pred)

    print("MSE con libreria del Random Forest: ",error)

