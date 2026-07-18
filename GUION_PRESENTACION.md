# Guion de Presentacion — Capstone

## Segmentacion y Clasificacion de Jugadores para Personalizacion y Monetizacion

---

## 1. Apertura (1 min)

**"Buenos dias profesores. Hoy les presento mi proyecto de capstone, cuyo objetivo
es identificar y clasificar perfiles de jugadores para personalizar la experiencia
de juego y optimizar la monetizacion."**

**Problema:** "En la industria gaming actual, un enfoque generico para todos los
jugadores desperdicia oportunidades de engagement y de ingresos. No es lo mismo
un jugador casual que juega 10 minutos al dia que un 'whale' que gasta
semanalmente en contenido premium."

**Solucion:** "Construi un sistema basado en datos que segmenta automaticamente a
los jugadores y predice su nivel de engagement, permitiendo disenar estrategias
especificas para cada grupo."

---

## 2. Dataset (1 min)

- **40,034 registros**, 13 columnas
- Kaggle: *Predict Online Gaming Behavior Dataset*
- Variables clave:
  - `EngagementLevel` (target): Low / Medium / High
  - `PlayTimeHours`, `SessionsPerWeek`, `AvgSessionDurationMinutes`
  - `InGamePurchases` (compra in-game: si/no)
  - `PlayerLevel`, `AchievementsUnlocked`
  - `GameGenre`, `Gender`, `Age`, `Location`, `GameDifficulty`

**Slide clave:** mostrar las 3 primeras filas del dataset.

---

## 3. EDA — Hallazgos Principales (2 min)

**Distribucion del target:**
- Medium: 48.4%
- High: 25.8%
- Low: 25.8%

*"El dataset esta desbalanceado hacia engagement medio, pero tenemos
representacion suficiente de las 3 clases."*

**Relaciones que encontramos:**
- A mayor `PlayTimeHours` y `AchievementsUnlocked`, mayor engagement
- `PlayerLevel` correlaciona fuertemente con `AchievementsUnlocked` (0.71)
- No hay valores nulos — datos limpios
- Los jugadores con `InGamePurchases = 1` tienden a engagement mas alto

**Mostrar:** boxplots de horas de juego vs engagement, matriz de correlacion.

---

## 4. Segmentacion No Supervisada — K-Means (2 min)

*"Aplicamos K-Means con k=3 sobre las variables de comportamiento para descubrir
segmentos naturales de jugadores."*

| Cluster | Perfil | Caracteristicas |
|---------|--------|----------------|
| **0** | **Casual** | Bajo tiempo de juego, pocas sesiones, pocos logros |
| **1** | **Achiever** | Muchas horas, muchos logros, nivel alto |
| **2** | **Spender** | Altas compras in-game, sesiones regulares |

**Mostrar:** grafico PCA de clusters.

**Interpretacion de negocio:**
- **Casuales** -> retener con recompensas diarias y anuncios
- **Achievers** -> motivar con logros, tablas de clasificacion, battle pass
- **Spenders** -> monetizar con ofertas VIP y bundles premium

---

## 5. Modelo Baseline — Random Forest (1.5 min)

*"Como baseline entrenamos un Random Forest con 100 arboles."*

| Metric | Value |
|--------|-------|
| Accuracy | **91.16%** |
| Precision (macro avg) | 0.91 |
| Recall (macro avg) | 0.91 |
| F1-score (macro avg) | 0.91 |

**Feature Importance top 5:**
1. `PlayerLevel`
2. `AchievementsUnlocked`
3. `SessionsPerWeek`
4. `PlayTimeHours`
5. `AvgSessionDurationMinutes`

*"Esto nos dice que lo que mas determina el engagement es el nivel del jugador
y los logros desbloqueados, no tanto datos demograficos como edad o genero."*

**Mostrar:** matriz de confusion y grafico de importancia.

---

## 6. Red Neuronal Densa — MLP (2 min)

*"Para mejorar el baseline, implementamos una red neuronal con TensorFlow/Keras."*

**Arquitectura:**
```
Input (20 features) -> Dense(64, ReLU) + Dropout(20%)
                    -> Dense(32, ReLU) + Dropout(20%)
                    -> Dense(3, Softmax)
```

**Resultado:**

| Modelo | Accuracy |
|--------|----------|
| Random Forest | 91.16% |
| **MLP (Red Neuronal)** | **91.56%** |

*"La red neuronal supera ligeramente al Random Forest, y ademas nos da
probabilidades por clase — util para sistemas de recomendacion donde no
solo importa la clase, sino tambien el nivel de confianza."*

**Ejemplo de prediccion probabilistica:**
- Low: 2.86%
- Medium: 85.07%
- High: 12.06%
- -> Clase asignada: Medium

**Mostrar:** curvas de perdida/accuracy y matriz de confusion.

---

## 7. Estrategia de Monetizacion (2 min)

*"Con los segmentos identificados, proponemos las siguientes estrategias:"*

| Segmento | Engagement | Monetizacion |
|----------|-----------|--------------|
| **Casual** | Notificaciones push, recompensas diarias, misiones cortas | Anuncios intersticiales, starter packs ($1-5) |
| **Achiever** | Logros semanales, tablas de clasificacion, contenido desbloqueable | Pase de batalla ($10/mes), cosmeticos exclusivos |
| **Spender** | Ofertas VIP, acceso anticipado, contenido premium | Bundles con descuento, suscripcion premium ($20+/mes) |

**KPI esperados:**
- Aumento de retention rate en 15-20%
- Incremento de ARPU (Average Revenue Per User) en 25-30%
- Mejora en conversion de free-to-play a paying users

*"Este enfoque permite pasar de una estrategia 'one-size-fits-all' a una
personalizada por segmento, maximizando tanto la experiencia del usuario
como los ingresos."*

---

## 8. Pipeline Tecnico (1 min)

**Stack utilizado:**

| Herramienta | Proposito |
|-------------|-----------|
| Python + Pandas | Procesamiento de datos |
| Matplotlib + Seaborn | Visualizacion |
| Scikit-learn | K-Means, Random Forest, metricas, PCA |
| TensorFlow / Keras | Red Neuronal Densa |
| Jupyter Notebook | Prototipado y presentacion |

**Flujo:**
```
CSV -> EDA -> Preprocesamiento -> Clustering -> Random Forest -> MLP -> Exportacion
```

---

## 9. Mejoras Futuras (1 min)

*"El proyecto se puede escalar con estas mejoras:"*

1. **Feature engineering avanzado:** ratio logros/nivel, tiempo semanal total
2. **Modelos mas potentes:** XGBoost, LightGBM, TabTransformer
3. **MLOps:** API con FastAPI, dashboard en Streamlit/Power BI
4. **Sistema de recomendacion:** sugerir juegos/generos segun perfil
5. **A/B Testing:** validar que las campanas mejoran KPIs reales
6. **Deteccion de anomalias:** Autoencoders para identificar cambios de comportamiento

---

## 10. Cierre (30 seg)

*"En conclusion, este proyecto demuestra que con datos de comportamiento
basico podemos segmentar jugadores, predecir su engagement con ~92% de
precision y disenar estrategias de monetizacion especificas para cada perfil.
Esto se traduce directamente en mejor experiencia de usuario y mayores
ingresos para la plataforma."*

*"Gracias por su atencion. ?Preguntas?"*

---

## Preguntas Frecuentes (FAQ para la defensa)

**?Por que k=3 en K-Means?**
Evaluamos del 2 al 8 con metodo del codo y silhouette score. k=3 ofrece
el mejor balance entre interpretabilidad y calidad de clustering.

**?Por que MLP y no CNN/RNN?**
Los datos son tabulares, no secuenciales ni de imagen. Una red densa es
la arquitectura adecuada para este tipo de datos. Para mejorar, se podria
usar TabTransformer.

**?Por que el Silhouette Score es bajo (0.196)?**
Los clusters en comportamiento humano no son perfectamente esfericos ni
separados. El valor es aceptable para datos de consumo real y los perfiles
son interpretables y accionables.

**?Se puede usar en tiempo real?**
Si. El modelo MLP pesa ~300KB y una prediccion toma <1ms. Se puede
servir via API REST con FastAPI para integrarlo en cualquier plataforma.

**?Que pasa si llegan nuevos datos?**
El pipeline esta disenado para re-entrenamiento periodico. Se guarda el
scaler y el modelo, y con `model.fit()` en los nuevos datos se actualiza.
