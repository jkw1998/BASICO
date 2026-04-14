import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- PASO 1: DATOS DEL LIBRO (VPP Ejemplo Cap 8) ---
CAP_MAX = 50.0      # Capacidad batería (MWh)
P_MAX = 15.0        # Potencia máx (MW)
EFICIENCIA = 0.9    # η (90%)
SOC_INI = 25.0      # Estado inicial

# --- PASO 2: EL OPTIMIZADOR (Referencia matemática) ---
def resolver_ejemplo_libro(demanda, renovable):
    n = len(demanda)
    dem_neta = demanda - renovable
    
    # Objetivo: Minimizar el desajuste (Balance de la VPP)
    def obj(x): return np.sum((dem_neta + x)**2)

    # Restricciones de la sección 8.2.3
    def const_soc(x):
        soc = SOC_INI
        for val in x:
            if val >= 0: soc += val * EFICIENCIA
            else: soc += val / EFICIENCIA
            if soc < 0 or soc > CAP_MAX: return -1 # Castigo si rompe límites
        return 0

    res = minimize(obj, np.zeros(n), bounds=[(-P_MAX, P_MAX)]*n)
    
    # Calcular evolución del almacenamiento
    soc_list = []
    s = SOC_INI
    for v in res.x:
        if v >= 0: s += v * EFICIENCIA
        else: s += v / EFICIENCIA
        soc_list.append(s)
    return soc_list

# --- PASO 3: GENERAR DATASET Y ENTRENAR LA RED ---
print("1. Resolviendo ejemplo del libro para crear datos...")
datos = []
for _ in range(100): # 100 días de práctica
    d = np.random.normal(60, 10, 24) # Demanda
    r = np.random.normal(40, 15, 24) # Renovable
    soc_optimo = resolver_ejemplo_libro(d, r)
    for h in range(24):
        datos.append({'dem': d[h], 'ren': r[h], 'soc_ant': soc_optimo[h-1] if h>0 else SOC_INI, 'target': soc_optimo[h]})

df = pd.DataFrame(datos)
scaler = StandardScaler()
X = scaler.fit_transform(df[['dem', 'ren', 'soc_ant']])
y = df['target']

print("2. Entrenando la Red Neuronal para que aprenda la solución óptima...")
# Esta es la Red Neuronal (MLP = Multi-Layer Perceptron)
red_neuronal = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500)
red_neuronal.fit(X, y)

# --- PASO 4: RESULTADOS PARA MOSTRAR AL TUTOR ---
print("\n--- INFORME PARA EL TUTOR ---")
test_input = scaler.transform([[80, 10, 25]]) # Caso: Demanda 80, Renovable 10, SOC 25
pred = red_neuronal.predict(test_input)
print(f"Resultado: Ante falta de energía, la Red predice un SOC de {pred[0]:.2f} MWh")

plt.plot(red_neuronal.loss_curve_)
plt.title("Curva de Aprendizaje de la Red (Debería bajar)")
plt.ylabel("Error"); plt.xlabel("Iteraciones")
plt.show()