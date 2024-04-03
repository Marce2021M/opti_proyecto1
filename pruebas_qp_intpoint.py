from qp_intpoint import qp_intpoint
import numpy as np
import pandas as pd

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
    cols = x_.shape[1]

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
    # regresamos la solucion
    return qp_intpoint(Q, c, F, d)
    
def grafica_resultados(M_, B_, tipo):
    values_M = np.dot(M_, w_scaled) + b_scaled
    values_B = np.dot(B_, w_scaled) + b_scaled

    indices_M = np.arange(1, len(values_M) + 1)
    indices_B = np.arange(1, len(values_B) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot for matrix M
    plt.scatter(indices_M, values_M, color='blue', label='w*x + b (para M)')

    # Plot for matrix B
    plt.scatter(indices_B, values_B, color='red', label='w*x + b (para B)')

    plt.title(f'Separacion de los conjuntos ({tipo})')
    plt.xlabel('ID de x')
    plt.ylabel('Valor w*x + b')
    plt.legend()
    plt.grid(True)
    plt.show()

np.random.seed(42)
# Create the first dataframe with one negative feature and one random
df1 = pd.DataFrame({
    'Feature1': -np.abs(1+np.random.randn(200)),
    'Feature2': np.random.randn(200)
})

# Create the second dataframe with all positive first feature and one random
df2 = pd.DataFrame({
    'Feature1': np.abs(1+np.random.randn(200)),
    'Feature2': np.random.randn(200)
})

# Concatenate the two dataframes and create the target series
x = pd.concat([df1, df2], ignore_index=True)
y = pd.Series(['M']*200 + ['B']*200)
y.name = "Diagnosis"


xsol_train, mu_train, iter_ = separa(x, y)
b_scaled = xsol_train[-1,-1]
w_scaled = xsol_train[:-1, :]
print(w_scaled, b_scaled)
# checamos que cumplan las restricciones
M_train, B_train = obten_M_B(x, y)
values_M_train = np.dot(M_train, w_scaled) + b_scaled
values_B_train = np.dot(B_train, w_scaled) + b_scaled

# checa que los M sean menores a -1
assert np.all(values_M_train + 1 <= 1e-4), "No todos los valores M son menores a -1"
# checa que los B sean mayores a a
assert np.all(values_B_train - 1 >= -1e-4), "No todos los valores son mayores a 1"
print("Los valores de entrenamiento cumplen las restricciones")

## Graficamos los resultados
grafica_resultados(M_train, B_train, tipo="entrenamiento")