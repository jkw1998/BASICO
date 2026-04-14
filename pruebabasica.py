import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# ==========================================
# 1. PARÁMETROS DEL SISTEMA (EQUILIBRADOS)
# ==========================================
CAPACIDAD_MAX = 150.0  # Aumentada para soportar ciclos nocturnos
P_MAX_BAT = 30.0       # Potencia de carga/descarga
SOC_INICIAL = 75.0     # Empezamos con reserva
EFICIENCIA = 0.95

# ==========================================
# 2. GENERACIÓN DE DATOS DE ENTRENAMIENTO
# ==========================================
# Entrenamos a la IA para que aprenda a COMPENSAR el balance neto
print(" FASE 1: ENTRENAMIENTO DE LA RED NEURONAL ")
X_train, y_train = [], []

for _ in range(600):  # Generamos diversidad de escenarios
    h = np.arange(24)
    # Escenarios donde el sol y la demanda están más pareados
    dem = 20 + 10 * np.sin(2*np.pi*h/24) + np.random.normal(0, 2, 24)
    ren = 50 * np.exp(-((h-12)**2)/8) if np.random.rand() > 0.2 else np.zeros(24)
    
    soc_t = SOC_INICIAL
    for i in range(24):
        balance_neta = dem[i] - ren[i]
        # La "acción maestra" es intentar cubrir el hueco
        accion_ideal = -balance_neta * 0.9 
        accion_ideal = np.clip(accion_ideal, -P_MAX_BAT, P_MAX_BAT)
        
        # Validar límites físicos del SOC
        if soc_t + accion_ideal > CAPACIDAD_MAX: accion_ideal = CAPACIDAD_MAX - soc_t
        if soc_t + accion_ideal < 0: accion_ideal = -soc_t
        
        X_train.append([dem[i], ren[i], soc_t])
        y_train.append(accion_ideal)
        soc_t += accion_ideal

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Red Neuronal robusta
modelo_vpp = MLPRegressor(hidden_layer_sizes=(64, 64, 32), 
                          activation='relu', 
                          max_iter=3000, 
                          random_state=1)
modelo_vpp.fit(X_scaled, y_train)
print("Modelo entrenado con exito!")

# ==========================================
# 3. SIMULACIÓN SEMANAL (168 HORAS)
# ==========================================
print("\n FASE 2: SIMULACIÓN DE 7 DÍAS ")
warnings.filterwarnings("ignore")
soc_actual = SOC_INICIAL
resultados = []

for h in range(168):
    # --- ENTRADAS DINÁMICAS (Aquí puedes modificar para testear) ---
    # Demanda con pico a las 19:00 (h%24 == 19)
    dem_h = 25 + 12 * np.sin(2 * np.pi * (h % 24 - 10) / 24)
    
    # Ciclo solar: Día sí, día con poco sol, día nublado...
    dia_num = (h // 24) % 7
    if dia_num in [0, 2, 4, 6]: # Días soleados
        ren_h = 65 * np.exp(-((h % 24 - 12)**2) / 6)
    elif dia_num in [1, 3]:     # Días nublados
        ren_h = 20 * np.exp(-((h % 24 - 12)**2) / 6)
    else:                       # Día de tormenta (cero sol)
        ren_h = 0
    
    if ren_h < 0.5: ren_h = 0

    # Predicción de la Acción (IA)
    input_scaled = scaler.transform([[dem_h, ren_h, soc_actual]])
    accion_ia = modelo_vpp.predict(input_scaled)[0]
    
    # Post-procesamiento físico (Seguridad)
    if soc_actual + accion_ia > CAPACIDAD_MAX: accion_ia = CAPACIDAD_MAX - soc_actual
    if soc_actual + accion_ia < 0: accion_ia = -soc_actual
    
    soc_actual += accion_ia
    resultados.append([h, dem_h, ren_h, dem_h - ren_h, soc_actual, accion_ia])

df = pd.DataFrame(resultados, columns=['H', 'Dem', 'Ren', 'Neta', 'SOC', 'Accion'])

# ==========================================
# 4. VISUALIZACIÓN PROFESIONAL
# ==========================================
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, 
                               gridspec_kw={'height_ratios': [1, 1.2]})

# --- GRÁFICA SUPERIOR: BALANCE ENERGÉTICO ---
ax1.plot(df['H'], df['Dem'], color='red', label='Demanda (Consumo)', linewidth=1.5)
ax1.plot(df['H'], df['Ren'], color='green', label='Renovable (Generación)', linewidth=1.5)
ax1.fill_between(df['H'], df['Neta'], 0, where=(df['Neta'] > 0), color='red', alpha=0.15, label='Déficit')
ax1.fill_between(df['H'], df['Neta'], 0, where=(df['Neta'] < 0), color='green', alpha=0.15, label='Exceso')
ax1.set_ylabel('Potencia (MW)', fontweight='bold')
ax1.set_title('ENTRADAS DEL SISTEMA Y BALANCE NETO', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', ncol=2, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)

# --- GRÁFICA INFERIOR: GESTIÓN DE BATERÍA ---
ax2.plot(df['H'], df['SOC'], color='mediumblue', linewidth=3, label='Estado de Carga (SOC)')
ax2.bar(df['H'], df['Accion'], color='purple', alpha=0.5, label='Flujo Batería (IA)')

ax2.set_xlabel('Tiempo (Horas de la semana)', fontweight='bold')
ax2.set_ylabel('Energía (MWh) / Potencia (MW)', fontweight='bold')
ax2.set_ylim(-10, CAPACIDAD_MAX + 30)
ax2.set_title('RESPUESTA DE LA RED NEURONAL (GESTIÓN VPP)', fontsize=14, fontweight='bold')

# Líneas de referencia
ax2.axhline(y=CAPACIDAD_MAX, color='black', linestyle='--', linewidth=1.5, label='Capacidad Máxima')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Marcas de días (cada 24h)
for i in range(0, 169, 24):
    ax2.axvline(x=i, color='gray', linestyle=':', alpha=0.5)

ax2.legend(loc='upper right', ncol=2, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)

plt.xticks(np.arange(0, 169, 24))
plt.tight_layout()
plt.show()

# Resumen numérico para validar
print("\nResumen de gestion (Dia 1):")
print(df[['H', 'Neta', 'SOC', 'Accion']].head(24).to_string(index=False))