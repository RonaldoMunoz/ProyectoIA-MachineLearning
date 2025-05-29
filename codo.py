import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Cargar el dataset (ya lo hiciste en dataset.py, pero lo ponemos aquí completo)
df = pd.read_csv("./Pokemon_processed.csv")

# Selección de características relevantes
features = ['Agresividad', 'Resistencia', 'Movilidad', 'Especialista ofensivo', 'Especialista defensivo', 'Balanceado']
X = df[features]

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar el número óptimo de clústeres con el método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método del codo')
plt.xlabel('Número de clústeres')
plt.ylabel('WCSS')
plt.grid()
plt.show()
