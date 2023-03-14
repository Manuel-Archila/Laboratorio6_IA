# Task 1.1

- Métrica de desempeño

    Se utilizó accuracy ya que al analizar las variables se determinó que si estaba balanceado el dataset. 
    
- ¿Qué métrica usaron para seleccionar los features?

    Para seleccionar los features de este dataset tomamos en cuenta las varibales que tenian una correlación mayor a 0.1 o menor a -0.1 con la varibale objetivo,
    estas son las que tienen una relación directamente o inversamente proporcional a la variable.

- ¿Experimentaron overfitting?

    No se experimentó overfitting

- Se realizó una normalización de las variables que se iban a tomar en cuenta en el modelo. 

- Al realizar el modelo tanto con nuestra implementación como con la librería se encontraron 

- Comparacion de modelos

    El modelo que mejor realiza las predicciones es el modelo de la libreria que se puede ver reflejado en el accuracy del modelo con un 0.71 mayor al 0.70 de nuestro modelo, esto se puede deber a que en esta implementacion esta optimizada y utiliza algoritmos mas avanzados para realizar las predicciones.

- Se implemento random forest en este task con el fin de probar como cambiarian los resultados con respecto a los modelos elaborados anteriormente.

# Task 1.2

- Métrica de desempeño

    Como utilizamos un arbol de regresión y el dataset no estaba balanceado optamos por usar la métrica MSE
    
- ¿Qué métrica usaron para seleccionar los features?

    Para seleccionar los features de este dataset tomamos en cuenta las varibales que tenian una correlación mayor a 0.1 o menor a -0.1 con la varibale objetivo,
    estas son las que tienen una relación directamente o inversamente proporcional a la variable.

- ¿Experi se removieron variables como el ID de cada jugador que solo harían ruido en el entrenamiento.en remoron overfitting?

    No se experimentó overfitting

- En cuanto a tuning se realizo una normalización de las variables que se iban a tomar en cuenta en el modelo. 

- Se implemento random forest en este task con el fin de probar como cambiarian los resultados con respecto a los modelos elaborados anteriormente.