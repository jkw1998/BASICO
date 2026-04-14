import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- PARÁMETROS CAPÍTULO 8 ---
CAPACIDAD_MAX = 50.0
P_MAX_CARGA = 15.0
P_MAX_DESCARGA = 15.0
EFICIENCIA = 0.9
SOC_INICIAL = 25.0

# --- EL OPTIMIZADOR (MAESTRO) ---
def optimizador_vpp_cap8(demanda, renovable):
    n = len(demanda)
    demanda_neta = demanda - renovable
    bounds = [(-P_MAX_DESCARGA, P_MAX_CARGA) for _ in range(n)]

    def objetivo(x):
        return np.sum((demanda_neta + x)**2)

    def restriccion_soc(x):
        soc = np.zeros(n + 1)
        soc[0] = SOC_INICIAL
        penalizacion = 0
        for t in range(n):
            if x[t] >= 0: soc[t+1] = soc[t] + (x[t] * EFICIENCIA)
            else: soc[t+1] = soc[t] + (x[t] / EFICIENCIA)
            if soc[t+1] < 0: penalizacion += abs(soc[t+1])
            elif soc[t+1] > CAPACIDAD_MAX: penalizacion += (soc[t+1] - CAPACIDAD_MAX)
        return penalizacion

    res = minimize(objetivo, np.zeros(n), method='SLSQP', bounds=bounds, 
                   constraints={'type': 'eq', 'fun': restriccion_soc})
    
    soc_evol = []
    curr = SOC_INICIAL
    for val in res.x:
        if val >= 0: curr += (val * EFICIENCIA)
        else: curr += (val / EFICIENCIA)
        soc_evol.append(curr)
    return soc_evol

# --- GENERACIÓN DE DATOS ---
print("Generando datos...")
datos = []
for dia in range(200):
    base = np.array([30,25,20,20,25,35,50,60,70,75,80,75,70,65,70,80,90,100,95,85,70,55,45,35])
    demanda = base + np.random.normal(0, 5, 24)
    renov = np.random.uniform(5, 55, 24)
    niveles = optimizador_vpp_cap8(demanda, renov)
    for h in range(24):
        datos.append({'d': demanda[h], 'r': renov[h], 'sp': niveles[h-1] if h>0 else SOC_INICIAL, 'target': niveles[h]})

df = pd.DataFrame(datos)

# --- RED NEURONAL (CON SCIKIT-LEARN) ---
X = df[['d', 'r', 'sp']].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Entrenando Red Neuronal (MLP)...")
# Creamos la red con 2 capas ocultas (64 y 32 neuronas)
regr = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, activation='relu', solver='adam')
regr.fit(X_train_scaled, y_train)

# --- RESULTADOS ---
score = regr.score(X_test_scaled, y_test)
print(f"Precisión del modelo (R2): {score:.4f}")

# Prueba
test_in = scaler.transform([[80, 15, 20]])
pred = regr.predict(test_in)
print(f"Predicción para Demanda=80, Renov=15, SOC_prev=20: {pred[0]:.2f} MWh")

plt.plot(regr.loss_curve_)
plt.title("Curva de Aprendizaje")
plt.show()