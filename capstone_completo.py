"""
Capstone: Segmentación y Clasificación de Jugadores
Pipeline completo: EDA => Preprocesamiento => Clustering => Random Forest => MLP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, silhouette_score)
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'online_gaming_behavior_dataset.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}

# ============================================================
# FASE 1: EDA
# ============================================================
print("=" * 60)
print("FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas\n")
print("Primeras 5 filas:")
print(df.head(), "\n")
print("Info del dataset:")
df.info()
print("\nEstadísticas descriptivas:")
print(df.describe(include='all'), "\n")
print("Valores nulos por columna:")
print(df.isnull().sum(), "\n")

print("Distribución de EngagementLevel:")
print(df['EngagementLevel'].value_counts())
print(f"Proporción: {df['EngagementLevel'].value_counts(normalize=True)}\n")

print("Distribución de Género (Gender):")
print(df['Gender'].value_counts(), "\n")
print("Distribución de Género de Juego (GameGenre):")
print(df['GameGenre'].value_counts(), "\n")
print("Distribución de Dificultad (GameDifficulty):")
print(df['GameDifficulty'].value_counts(), "\n")
print("Distribución de Location:")
print(df['Location'].value_counts(), "\n")
print("Compras in-game (InGamePurchases):")
print(df['InGamePurchases'].value_counts(), "\n")

results['dimensiones'] = {'filas': int(df.shape[0]), 'columnas': int(df.shape[1])}
results['target_dist'] = df['EngagementLevel'].value_counts().to_dict()

# ---- Visualizaciones ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.countplot(data=df, x='EngagementLevel', order=['Low', 'Medium', 'High'],
              palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Distribución de EngagementLevel')

sns.countplot(data=df, x='Gender', palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Distribución por Género')

sns.countplot(data=df, x='GameGenre', palette='Set3', ax=axes[0, 2])
axes[0, 2].set_title('Distribución por Género de Juego')
axes[0, 2].tick_params(axis='x', rotation=45)

sns.histplot(data=df, x='Age', bins=30, kde=True, color='steelblue', ax=axes[1, 0])
axes[1, 0].set_title('Distribución de Edad')

sns.histplot(data=df, x='PlayTimeHours', bins=30, kde=True, color='coral', ax=axes[1, 1])
axes[1, 1].set_title('Distribución de Horas de Juego')

sns.countplot(data=df, x='InGamePurchases', palette='Set1', ax=axes[1, 2])
axes[1, 2].set_title('Compras In-Game (0=No, 1=Sí)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_distribuciones.png'), dpi=150)
plt.close()
print("[OK] Grafico de distribuciones guardado\n")

# Boxplots por EngagementLevel
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

num_cols = ['Age', 'PlayTimeHours', 'SessionsPerWeek',
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
titles = ['Edad', 'Horas de Juego', 'Sesiones por Semana',
          'Duración Promedio Sesión', 'Nivel del Jugador', 'Logros Desbloqueados']

for ax, col, title in zip(axes.flat, num_cols, titles):
    sns.boxplot(data=df, x='EngagementLevel', y=col,
                order=['Low', 'Medium', 'High'], palette='Set2', ax=ax)
    ax.set_title(f'{title} vs EngagementLevel')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_boxplots_engagement.png'), dpi=150)
plt.close()
print("[OK] Boxplots por engagement guardados\n")

# Matriz de correlación
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5)
plt.title('Matriz de Correlación - Variables Numéricas')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_matriz_correlacion.png'), dpi=150)
plt.close()
print("[OK] Matriz de correlacion guardada\n")

results['correlaciones'] = corr.to_dict()

# ============================================================
# FASE 2: PREPROCESAMIENTO
# ============================================================
print("=" * 60)
print("FASE 2: PREPROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS")
print("=" * 60)

df_clean = df.drop(columns=['PlayerID'])

categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

target_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_encoded['EngagementLevel'] = df_encoded['EngagementLevel'].map(target_map)

X = df_encoded.drop(columns=['EngagementLevel'])
y = df_encoded['EngagementLevel']

feature_names = X.columns.tolist()
print(f"Características después de One-Hot: {X.shape[1]}")
print(f"Columnas: {feature_names}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}\n")

# Escalado para Red Neuronal
scaler_nn = MinMaxScaler()
X_train_scaled = scaler_nn.fit_transform(X_train)
X_test_scaled = scaler_nn.transform(X_test)

# Escalado Standard para clustering
scaler_std = StandardScaler()
# Características para clustering
cluster_feats = ['PlayTimeHours', 'SessionsPerWeek',
                 'AvgSessionDurationMinutes', 'InGamePurchases', 'AchievementsUnlocked']
X_clust = scaler_std.fit_transform(df_clean[cluster_feats])

results['preprocessing'] = {
    'n_features': X.shape[1],
    'feature_names': feature_names,
    'train_size': X_train.shape[0],
    'test_size': X_test.shape[0],
    'target_map': target_map
}

# ============================================================
# FASE 3: SEGMENTACIÓN NO SUPERVISADA (K-MEANS)
# ============================================================
print("=" * 60)
print("FASE 3: SEGMENTACIÓN NO SUPERVISADA (K-Means)")
print("=" * 60)

# Determinar k óptimo con el método del codo
inertias = []
silhouettes = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_clust, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('Método del Codo')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inercia')
axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_optimo_k.png'), dpi=150)
plt.close()
print("[OK] Grafico de optimizacion de k guardado\n")

k_optimo = 3
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_clust)

cluster_profile = df_clean.groupby('Cluster')[cluster_feats].mean()
cluster_profile['Count'] = df_clean['Cluster'].value_counts()
print("Perfiles de Clusters:")
print(cluster_profile, "\n")

# Interpretación de clusters basada en perfiles
cluster_means = df_clean.groupby('Cluster')[cluster_feats].mean()
cluster_labels = {}
for c in cluster_means.index:
    row = cluster_means.loc[c]
    if row['PlayTimeHours'] < cluster_means['PlayTimeHours'].median():
        label = "Casual"
    elif row['InGamePurchases'] > cluster_means['InGamePurchases'].median():
        label = "Spender"
    else:
        label = "Achiever"
    cluster_labels[int(c)] = label

print("Interpretación de Clusters:")
for c, label in sorted(cluster_labels.items()):
    print(f"  Cluster {c}: {label}")
print()

# PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clust)
df_clean['PCA1'] = X_pca[:, 0]
df_clean['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(data=df_clean, x='PCA1', y='PCA2',
                          hue='Cluster', palette='Set1', alpha=0.6)
plt.title('Visualización de Clusters (PCA)')
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_pca.png'), dpi=150)
plt.close()
print("[OK] Visualizacion PCA de clusters guardada\n")

results['clustering'] = {
    'k_optimo': k_optimo,
    'silhouette_score': float(silhouettes[k_optimo - 2]),
    'perfiles': cluster_profile.to_dict(),
    'interpretacion': cluster_labels
}

# ============================================================
# FASE 4: RANDOM FOREST (BASELINE)
# ============================================================
print("=" * 60)
print("FASE 4: CLASIFICACIÓN CON RANDOM FOREST (BASELINE)")
print("=" * 60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {acc_rf:.4f}\n")
print(classification_report(y_test, y_pred_rf, target_names=['Low', 'Medium', 'High']))

# Feature importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 características más importantes:")
print(importances.head(10), "\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d',
            cmap='Blues', xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'], ax=axes[0])
axes[0].set_title('Matriz de Confusión - Random Forest')
axes[0].set_xlabel('Predicho')
axes[0].set_ylabel('Real')

axes[1].barh(range(10), importances.head(10)['importance'][::-1])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(importances.head(10)['feature'][::-1])
axes[1].set_title('Top 10 Feature Importance')
axes[1].set_xlabel('Importancia')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_resultados.png'), dpi=150)
plt.close()
print("[OK] Graficos de Random Forest guardados\n")

results['random_forest'] = {
    'accuracy': float(acc_rf),
    'classification_report': classification_report(
        y_test, y_pred_rf, target_names=['Low', 'Medium', 'High'], output_dict=True
    ),
    'feature_importance': importances.head(10).to_dict()
}

# ============================================================
# FASE 5: RED NEURONAL DENSA (MLP)
# ============================================================
print("=" * 60)
print("FASE 5: CLASIFICACIÓN CON RED NEURONAL DENSA (MLP)")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred_mlp = np.argmax(y_pred_probs, axis=1)

acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\nAccuracy MLP: {acc_mlp:.4f}\n")
print("--- MÉTRICAS RED NEURONAL DENSA (MLP) ---")
print(classification_report(y_test, y_pred_mlp, target_names=['Low', 'Medium', 'High']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'], label='Entrenamiento')
axes[0].plot(history.history['val_loss'], label='Validación')
axes[0].set_title('Curva de Pérdida')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Entrenamiento')
axes[1].plot(history.history['val_accuracy'], label='Validación')
axes[1].set_title('Curva de Accuracy')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

sns.heatmap(confusion_matrix(y_test, y_pred_mlp), annot=True, fmt='d',
            cmap='Purples', xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'], ax=axes[2])
axes[2].set_title('Matriz de Confusión - MLP')
axes[2].set_xlabel('Predicho')
axes[2].set_ylabel('Real')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_resultados.png'), dpi=150)
plt.close()
print("[OK] Graficos de MLP guardados\n")

# Ejemplo de predicción probabilística
new_player = X_test_scaled[0:1]
probs = model.predict(new_player, verbose=0)[0]
classes = ['Low', 'Medium', 'High']
print("Ejemplo de predicción probabilística:")
for i, c in enumerate(classes):
    print(f"  {c}: {probs[i]*100:.2f}%")
print(f"  => Clase asignada: {classes[np.argmax(probs)]}\n")

results['mlp'] = {
    'accuracy': float(acc_mlp),
    'epochs_trained': len(history.history['loss']),
    'classification_report': classification_report(
        y_test, y_pred_mlp, target_names=['Low', 'Medium', 'High'], output_dict=True
    ),
    'ejemplo_prediccion': {c: float(probs[i]) for i, c in enumerate(classes)}
}

# Guardar modelo y scaler
model.save(os.path.join(OUTPUT_DIR, 'modelo_mlp.keras'))
import pickle
with open(os.path.join(OUTPUT_DIR, 'scaler_nn.pkl'), 'wb') as f:
    pickle.dump(scaler_nn, f)
with open(os.path.join(OUTPUT_DIR, 'scaler_cluster.pkl'), 'wb') as f:
    pickle.dump(scaler_std, f)
print("[OK] Modelo y scalers guardados en output/\n")

# Resumen final
print("=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)
print(f"Random Forest Accuracy:  {results['random_forest']['accuracy']:.4f}")
print(f"MLP (Red Neuronal) Accuracy: {results['mlp']['accuracy']:.4f}")
print(f"Silhouette Score (clustering): {results['clustering']['silhouette_score']:.4f}")

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Resultados exportados a {OUTPUT_DIR}")
print("!Pipeline completado exitosamente!")
