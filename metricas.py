def metrica(y_pred, y_true):
    # se usa accuracy porque los datos estan bien balanceados.

    true_postive = 0
    true_negative = 0
    positive = 0
    negative = 0

    y_true = y_true.tolist()

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            positive += 1
            if y_true[i] == 1:
                true_postive += 1
        else:
            negative += 1
            if y_true[i] == 0:
                true_negative += 1
    
    accuracy = (true_postive + true_negative) / (positive + negative)
    return accuracy