import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

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

    # # Se separa el dataset en training y testing
    copy = df

    X = copy.drop("blueWins", axis=1)
    y = df["blueWins"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



    pass