from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from qp_intpoint import qp_intpoint
import matplotlib.pyplot as plt

# Regresa dos matrices que representan una particion de los datos x
# una matriz corresponde a los que tienen y = M
# la otra a los que tienen y = B
def obten_M_B(x_, y_):
    z = pd.concat([x_, y_],axis = 1)

    rows = x_.shape[0]
    cols = y_.shape[1]

    z = pd.concat([x_, y_], axis=1)

    zB = z[z['Diagnosis'] == 'B']
    zM = z[z['Diagnosis'] == 'M']

    B = zB.iloc[:, :-1].to_numpy()
    M = zM.iloc[:, :-1].to_numpy()

    return M, B

def separa(x_, y_):
    rows = x_.shape[0]
    cols = x_.shape[1]

    M, B = obten_M_B(x_, y_)
    
    # Construimos las matrices para usar el metodo de puntos interiores
    F = np.block([[B, np.ones((B.shape[0], 1))], [-M, -np.ones((M.shape[0], 1))]])
    d = np.ones((rows,1))
    
    Q = np.eye(cols+1)
    Q[cols,cols] = 0
    c = np.zeros((cols+1,1))

    print(F.shape, d.shape, Q.shape, c.shape)
    # regresamos la solucion
    return qp_intpoint(Q, c, F, d)

def obten_cuantos_cumplen(M_, B_):
    M_, B_ = obten_M_B(x_test_scaled_reset, y_test_reset)
    values_M = np.dot(M_, w_scaled) + b_scaled
    values_B = np.dot(B_, w_scaled) + b_scaled

    # cuenta cuantos del test cumplen y no cumplen para la categoria M
    comply_M = np.sum((values_M + 1) <= 1e-4)
    not_comply_M = np.sum((values_M + 1) > 1e-4)

    # cuenta cuandos del test cumplen y no cumplen para la categoria B
    comply_B = np.sum((values_B - 1) >= -1e-4)
    not_comply_B = np.sum((values_B - 1) < -1e-4)
    return comply_M, not_comply_M, comply_B, not_comply_B

def obten_metricas(comply_M, not_comply_M, comply_B, not_comply_B):
    TP = comply_M  # M values that are correctly less than -1
    FN = not_comply_M  # M values that are incorrectly not less than -1
    TN = comply_B  # B values that are correctly greater than 1
    FP = not_comply_B  # B values that are incorrectly not greater than 1

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 score
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, F1
    

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
x = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# dividimos en conjunto de entrenamiento y de prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# creamos un escalador para rescalar los datos de entrenamiento
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns)

x_train_scaled_reset = x_train_scaled.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

xsol_train, mu_train, iter_ = separa(x_train_scaled_reset, y_train_reset)
b_scaled = xsol_train[-1,-1]
w_scaled = xsol_train[:-1, :]

# checamos que cumplan las restricciones
M_train, B_train = obten_M_B(x_train_scaled_reset, y_train_reset)
values_M_train = np.dot(M_train, w_scaled) + b_scaled
values_B_train = np.dot(B_train, w_scaled) + b_scaled

# checa que los M sean menores a -1
assert np.all(values_M_train + 1 <= 1e-4), "No todos los valores M son menores a -1"
# checa que los B sean mayores a a
assert np.all(values_B_train - 1 >= -1e-4), "No todos los valores son mayores a 1"
print("Los valores cumplen las restricciones")

#### Ahora probamos el conjunto de prueba
x_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns)
x_test_scaled_reset = x_test_scaled.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# Printing the results
M_test, B_test = obten_M_B(x_test_scaled_reset, y_test_reset)
comply_M, not_comply_M, comply_B, not_comply_B = obten_cuantos_cumplen(M_test, B_test)
print(f"Valores M que cumplen las restricciones: {comply_M}")
print(f"Valores M que no cumplen las restricciones: {not_comply_M}")
print(f"Valores B que cumplen las restricciones: {comply_B}")
print(f"Valores B que no cumplen las restricciones: {not_comply_B}")

## Ahora obtenemos metricas del modelo
accuracy, precision, recall, F1 = obten_metricas(comply_M, not_comply_M, comply_B, not_comply_B)
# Printing the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {F1:.4f}")
