# Guía del Proyecto Capstone: Segmentación y Clasificación de Jugadores

## 1. Descripción del Problema

**Objetivo del negocio:** Identificar y clasificar a los jugadores/usuarios de videojuegos para personalizar la experiencia de juego, aumentar el *engagement* y optimizar la monetización.

**Objetivo técnico:** Construir un sistema de ML/DL que segmente perfiles de usuario según su comportamiento, preferencias y patrones de consumo dentro de la plataforma.

---

## 2. Dataset: `online_gaming_behavior_dataset.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `PlayerID` | int | Identificador único del jugador |
| `Age` | int | Edad |
| `Gender` | categórica | Male / Female |
| `Location` | categórica | USA, Europe, Asia, Other |
| `GameGenre` | categórica | Action, RPG, Strategy, Simulation, Sports |
| `PlayTimeHours` | float | Horas de juego por sesión |
| `InGamePurchases` | int | 0 = No, 1 = Sí (ha realizado compras) |
| `GameDifficulty` | categórica | Easy, Medium, Hard |
| `SessionsPerWeek` | int | Sesiones por semana |
| `AvgSessionDurationMinutes` | int | Duración promedio de sesión en minutos |
| `PlayerLevel` | int | Nivel alcanzado |
| `AchievementsUnlocked` | int | Logros desbloqueados |
| `EngagementLevel` | categórica (target) | Low, Medium, High |

---

## 3. Metodología Propuesta

### Fase 1 — Análisis Exploratorio de Datos (EDA)

- Carga y exploración de dimensiones del dataset (~40K registros)
- Análisis de balance de la variable objetivo (`EngagementLevel`)
- Distribuciones univariadas: edad, horas de juego, sesiones, compras, logros
- Relaciones bivariadas: horas de juego vs engagement, compras vs engagement, género vs género de juego
- Matriz de correlación entre variables numéricas
- Detección de valores atípicos y missing values

### Fase 2 — Preprocesamiento e Ingeniería de Características

- Eliminación de `PlayerID` (identificador sin valor predictivo)
- One-Hot Encoding para variables categóricas: `Gender`, `Location`, `GameGenre`, `GameDifficulty`
- Mapeo ordinal de `EngagementLevel`: Low→0, Medium→1, High→2
- División train/test (80/20) con estratificación para mantener la proporción de clases
- Normalización con `MinMaxScaler` (óptimo para redes neuronales) o `StandardScaler`

### Fase 3 — Segmentación No Supervisada (Clustering)

**Algoritmo:** K-Means (k=3) basado en:
- `PlayTimeHours`, `SessionsPerWeek`, `AvgSessionDurationMinutes`
- `InGamePurchases`, `AchievementsUnlocked`

**Perfiles esperados:**

| Cluster | Perfil | Comportamiento |
|---|---|---|
| 0 | Casual / Ocassional | Bajo tiempo de juego, pocas sesiones, bajos logros |
| 1 | Achiever / Comprometido | Altas horas de juego, muchos logros desbloqueados |
| 2 | Spender / Monetizado | Altas compras in-game, sesiones regulares |

**Alternativas:** Probar con DBSCAN o Gaussian Mixture Models si los clusters no son esféricos.

### Fase 4 — Clasificación con ML Clásico (Baseline)

**Modelo:** Random Forest Classifier (100 árboles)

- Sirve como baseline por su robustez ante no-linealidades y outliers
- Evaluación: accuracy, precision, recall, f1-score, matriz de confusión
- **Feature importance:** permite identificar qué variables pesan más en el engagement

**Alternativas:** XGBoost / LightGBM / Gradient Boosting (mejor performance, más tuning).

### Fase 5 — Clasificación con Deep Learning (MLP)

**Arquitectura de Red Neuronal Densa:**

```
Input (n_features)
  ↓
Dense(64, ReLU) + Dropout(0.2)
  ↓
Dense(32, ReLU) + Dropout(0.2)
  ↓
Dense(3, Softmax) → salida: Low / Medium / High
```

- Compilación: Adam + sparse_categorical_crossentropy
- Early Stopping para evitar overfitting
- Evaluación con métricas y matriz de confusión
- Predicción probabilística (útil para sistemas de recomendación)

**Alternativas:** Redes más profundas, batch normalization, learning rate scheduling.

---

## 4. Resultados Esperados

| Modelo | Accuracy esperado |
|---|---|
| Random Forest | ~85–90% |
| MLP (Red Neuronal) | ~88–92% |
| XGBoost (optativo) | ~90–93% |

---

## 5. Estrategia de Monetización y Personalización

Basado en los clusters/segmentos identificados:

| Segmento | Estrategia de Engagement | Estrategia de Monetización |
|---|---|---|
| **Casuales** | Notificaciones push, recompensas diarias | Anuncios intersticiales, ofertas de entrada |
| **Achievers** | Logros semanales, tablas de clasificación | Pase de batalla, cosméticos exclusivos |
| **Spenders** | Ofertas VIP, contenido premium anticipado | Paquetes con descuento por volumen, bundles |

- **Personalización dinámica:** mostrar misiones, promociones y UI adaptada al perfil
- **Sistema de recomendación:** sugerir juegos/géneros según el perfil y historial
- **Retención predictiva:** detectar jugadores en riesgo de abandono (bajo engagement que baja) y activar campañas de re-engagement

---

## 6. Mejoras y Extensiones Posibles

- **Feature Engineering avanzado:**
  - Ratio `AchievementsUnlocked / PlayerLevel` (eficiencia del jugador)
  - `AvgSessionDurationMinutes * SessionsPerWeek` (tiempo semanal total)
  - Interacciones entre género del juego y género del jugador
- **Modelos alternativos:** XGBoost, LightGBM, SVM con kernel RBF
- **Deep Learning avanzado:**
  - Embeddings para las variables categóricas (en lugar de One-Hot)
  - Red con atención o arquitectura TabTransformer
  - Autoencoders para detección de anomalías (jugadores que se salen del perfil)
- **MLOps:**
  - Guardar el scaler y el modelo entrenado (`.pkl` / `.h5`)
  - API de inferencia con FastAPI
  - Dashboard en Streamlit/Power BI para monitoreo de segmentos
- **A/B Testing:** validar que las campañas personalizadas realmente mejoran KPIs

---

## 7. Stack Tecnológico

| Herramienta | Uso |
|---|---|
| Python 3 + Pandas | Procesamiento de datos |
| Matplotlib + Seaborn | Visualización |
| Scikit-learn | Clustering, Random Forest, métricas |
| TensorFlow / Keras | Red Neuronal Densa |
| Jupyter Notebook | Prototipado y presentación |

---

## 8. Referencia

Este proyecto se basa en el dataset público *"Predict Online Gaming Behavior Dataset"* de Kaggle (rabieelkharoua). La guía base de trabajo se encuentra en `Capstone_Base_Guide.ipynb`, que cubre el pipeline completo de EDA → preprocesamiento → clustering → Random Forest → MLP.
