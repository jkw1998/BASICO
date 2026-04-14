import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

# ==========================================
# 1. PARÁMETROS DEL CASO DE ESTUDIO
# ==========================================
CAPACIDAD_MAX = 50.0  # MWh
P_MAX_BAT = 20.0      # MW
EFICIENCIA = 0.95     
SOC_INICIAL = 25.0    

def optimizador_maestro(demanda, renovable, soc_inicio):
    """ Genera la solución ideal que la IA debe aprender """
    n = len(demanda)
    d_neta = demanda - renovable
    def objetivo(x): return np.sum((d_neta + x)**2)
    def restriccion(x):
        s = soc_inicio
        for v in x:
            if v >= 0: s += v * EFICIENCIA
            else: s += v / EFICIENCIA
            if s < 0 or s > CAPACIDAD_MAX: return -100
        return 0
    res = minimize(objetivo, np.zeros(n), bounds=[(-P_MAX_BAT, P_MAX_BAT)] * n,
                   constraints={'type': 'eq', 'fun': restriccion})
    soc_evol = []
    curr_s = soc_inicio
    for v in res.x:
        if v >= 0: curr_s += v * EFICIENCIA
        else: curr_s += v / EFICIENCIA
        soc_evol.append(curr_s)
    return soc_evol

# ==========================================
# 2. GENERACIÓN DE DATOS Y ENTRENAMIENTO
# ==========================================
print("--- FASE 1: ENTRENANDO EL MODELO ---")
datos = []
# Entrenamos con 300 días para que la IA sea "sabia"
for _ in range(300):
    h = np.arange(24)
    # Escenarios variados para que la red aprenda a cargar y descargar
    d = 30 + 20 * np.sin(2 * np.pi * h/24) + np.random.normal(0, 5, 24)
    r = 20 + 30 * np.random.rand() * np.exp(-((h-12)**2)/6)
    
    soc_ref = optimizador_maestro(d, r, SOC_INICIAL)
    for i in range(24):
        datos.append([d[i], r[i], soc_ref[i-1] if i > 0 else SOC_INICIAL, soc_ref[i]])

df_train = pd.DataFrame(datos, columns=['dem', 'ren', 'soc_ant', 'target'])
scaler = MinMaxScaler()
X = scaler.fit_transform(df_train[['dem', 'ren', 'soc_ant']])
y = df_train['target']

# Red neuronal más profunda para evitar que la gráfica salga plana
modelo_vpp = MLPRegressor(hidden_layer_sizes=(100, 100, 50), activation='tanh', 
                          max_iter=2000, random_state=1)
modelo_vpp.fit(X, y)
print("¡Entrenamiento finalizado!")

# ==========================================
# 3. SIMULACIÓN SEMANAL (7 DÍAS = 168 HORAS)
# ==========================================
print("\n--- FASE 2: SIMULACIÓN DE 7 DÍAS ---")
warnings.filterwarnings("ignore")
soc_actual = SOC_INICIAL
historial = []

for h in range(168): # 7 días
    
    # -------------------------------------------------------
    # ESPACIO PARA MODIFICAR DATOS DE ENTRADA
    # -------------------------------------------------------
    # Aquí puedes cambiar las fórmulas para probar escenarios:
    
    # Ejemplo: Pico de demanda tarde y noche
    dem_h = 35 + 15 * np.sin(2 * np.pi * (h % 24 - 10) / 24) 
    
    # Ejemplo: Generación solar (puedes poner 0 para probar la noche)
    ren_h = 40 * np.exp(-((h % 24 - 12)**2) / 8) 
    if ren_h < 1: ren_h = 0 # Forzamos cero fuera de horas de sol
    
    # -------------------------------------------------------
    
    # Ejecución de la Red Neuronal
    input_ia = scaler.transform([[dem_h, ren_h, soc_actual]])
    nuevo_soc = modelo_vpp.predict(input_ia)[0]
    
    # Filtro de seguridad física
    nuevo_soc = max(0, min(CAPACIDAD_MAX, nuevo_soc))
    
    # Cálculo de potencia de la batería (Acción)
    accion = nuevo_soc - soc_actual
    
    historial.append({
        'Hora': h,
        'Día': (h // 24) + 1,
        'Demanda': round(dem_h, 2),
        'Renovable': round(ren_h, 2),
        'SOC_Batería': round(nuevo_soc, 2),
        'Acción_MW': round(accion, 2)
    })
    soc_actual = nuevo_soc

# Convertir a tabla para ver los valores
df_res = pd.DataFrame(historial)

# ==========================================
# 4. SALIDA DE VALORES Y GRÁFICAS
# ==========================================

# Mostramos los primeros 2 días en la terminal para inspección
print("\nPrimeras 48 horas de operación:")
print(df_res[['Hora', 'Demanda', 'Renovable', 'SOC_Batería', 'Acción_MW']].head(48).to_string(index=False))

# Gráfica estructurada
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Gráfica de Potencias
ax1.fill_between(df_res['Hora'], df_res['Demanda'], color='red', alpha=0.2, label='Demanda (Consumo)')
ax1.fill_between(df_res['Hora'], df_res['Renovable'], color='green', alpha=0.2, label='Renovable (Producción)')
ax1.set_ylabel('MW')
ax1.set_title('Entradas del Sistema (7 Días)')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# Gráfica del Storage (SOC) y Acción
ax2.plot(df_res['Hora'], df_res['SOC_Batería'], color='blue', linewidth=2, label='Nivel de Batería (SOC)')
ax2.bar(df_res['Hora'], df_res['Acción_MW'], color='purple', alpha=0.5, label='Carga (+) / Descarga (-)')
ax2.axhline(y=CAPACIDAD_MAX, color='black', linestyle='--', label='Límite Max')
ax2.axhline(y=0, color='black', linestyle='--', label='Límite Min')
ax2.set_ylabel('MWh / MW')
ax2.set_xlabel('Horas de la semana')
ax2.set_title('Respuesta de la Red Neuronal (Gestión de Almacenamiento)')
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()