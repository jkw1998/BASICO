import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PARÁMETROS TÉCNICOS (CAPÍTULO 8)
# ==========================================
CAPACIDAD_MAX = 50.0  # MWh
P_MAX_BAT = 15.0      # MW (Carga/Descarga)
EFICIENCIA = 0.9      # n (90%)
SOC_INICIAL = 25.0    # MWh (Punto de partida)

# ==========================================
# 2. OPTIMIZADOR DE REFERENCIA (BASADO EN EL LIBRO)
# ==========================================
def calcular_soc_optimo(demanda, renovable, soc_inicio):
    """
    Resuelve la operación del almacenamiento para 24h siguiendo
    las restricciones de las secciones 8.2 y 8.3.
    """
    n = len(demanda)
    demanda_neta = demanda - renovable
    
    # Función objetivo: Minimizar el desbalance de la VPP
    def objetivo(x):
        return np.sum((demanda_neta + x)**2)

    # Restricción de evolución del SOC (Ecuación 8.16 y 8.17)
    def restriccion_fisica(x):
        soc_temporal = soc_inicio
        for t in range(n):
            if x[t] >= 0: # Carga
                soc_temporal += x[t] * EFICIENCIA
            else:         # Descarga
                soc_temporal += x[t] / EFICIENCIA
            
            # Si el SOC sale de límites, devolvemos un valor de error
            if soc_temporal < 0 or soc_temporal > CAPACIDAD_MAX:
                return -1 
        return 0

    # Límites de potencia (Ecuación 8.18)
    bounds = [(-P_MAX_BAT, P_MAX_BAT)] * n
    
    res = minimize(objetivo, np.zeros(n), bounds=bounds, 
                   constraints={'type': 'eq', 'fun': restriccion_fisica})
    
    # Reconstruir la serie de SOC resultante
    soc_evolucion = []
    s = soc_inicio
    for v in res.x:
        if v >= 0: s += v * EFICIENCIA
        else: s += v / EFICIENCIA
        soc_evolucion.append(s)
        
    return res.x, soc_evolucion

# ==========================================
# 3. GENERACIÓN DE DATASET DE ENTRENAMIENTO
# ==========================================
print("Generando datos de entrenamiento...")
datos_historicos = []
num_dias_entrenamiento = 100

for _ in range(num_dias_entrenamiento):
    # Generar perfiles aleatorios de demanda y renovable
    d = np.random.normal(60, 10, 24)
    r = np.random.normal(40, 15, 24)
    _, soc_opt = calcular_soc_optimo(d, r, SOC_INICIAL)
    
    for h in range(24):
        datos_historicos.append({
            'demanda': d[h],
            'renovable': r[h],
            'soc_anterior': soc_opt[h-1] if h > 0 else SOC_INICIAL,
            'target_soc': soc_opt[h]
        })

df = pd.DataFrame(datos_historicos)

# Preparar IA
scaler = StandardScaler()
X = scaler.fit_transform(df[['demanda', 'renovable', 'soc_anterior']])
y = df['target_soc']

# Configurar y entrenar la Red Neuronal
modelo_ia = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
modelo_ia.fit(X, y)

# ==========================================
# 4. SIMULACIÓN DE 1 SEMANA (7 DÍAS)
# ==========================================
print("\nSimulando comportamiento para 1 semana...")
dias_simulacion = 7
soc_actual = SOC_INICIAL
resultados_sim = []

for h_total in range(24 * dias_simulacion):
    # Escenario aleatorio para la hora actual
    dem_h = np.random.normal(65, 12)
    ren_h = np.random.normal(35, 18)
    
    # Predicción de la IA
    input_ia = scaler.transform([[dem_h, ren_h, soc_actual]])
    nuevo_soc = modelo_ia.predict(input_ia)[0]
    
    # Asegurar límites físicos por seguridad (Post-procesado)
    nuevo_soc = max(0, min(CAPACIDAD_MAX, nuevo_soc))
    
    resultados_sim.append({
        'Hora': h_total,
        'Demanda': dem_h,
        'Renovable': ren_h,
        'SOC': nuevo_soc
    })
    soc_actual = nuevo_soc

# ==========================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ==========================================
res_df = pd.DataFrame(resultados_sim)

plt.figure(figsize=(15, 6))
plt.plot(res_df['Hora'], res_df['SOC'], label='Nivel Almacenamiento (SOC)', color='blue')
plt.bar(res_df['Hora'], res_df['Demanda'] - res_df['Renovable'], alpha=0.3, label='Demanda Neta', color='orange')
plt.axhline(y=CAPACIDAD_MAX, color='r', linestyle='--', label='Capacidad Máxima')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Simulación Semanal: Gestión de Almacenamiento mediante Red Neuronal')
plt.xlabel('Horas de la semana')
plt.ylabel('Energía (MWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()