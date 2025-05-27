import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Cargar el dataset
df = pd.read_csv("./Pokemon.csv")

# Selección de características relevantes
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X = df[features]

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar KMeans con 4 clústeres
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Mostrar cuántos Pokémon hay en cada clúster
print("Cantidad por clúster:")
print(df['Cluster'].value_counts())

# Estadísticas promedio por clúster
mean_stats = df.groupby('Cluster')[features].mean()
print("\nEstadísticas promedio por clúster:")
print(mean_stats)

# Nombres temáticos para los clanes (puedes cambiarlos)
clan_names = {
    0: "Clan Origen",      # Básicos, comunes, balanceados
    1: "Clan Relámpago",   # Rápidos y agresivos
    2: "Clan Celestial",   # Los más poderosos, legendarios
    3: "Clan Escudo"       # Defensivos, resistentes
}

# Asignar nombres al DataFrame
df['Clan'] = df['Cluster'].map(clan_names)

# PCA con 3 componentes para visualización 3D
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca_3d[:, 0]
df['PCA2'] = X_pca_3d[:, 1]
df['PCA3'] = X_pca_3d[:, 2]

# Visualización 3D interactiva con Plotly
fig = px.scatter_3d(
    df,
    x='PCA1',
    y='PCA2',
    z='PCA3',
    color='Clan',
    hover_name='Name',        # Muestra el nombre del Pokémon al pasar el mouse
    title='Pokémon Clustering 3D con PCA y Clanes Ancestrales',
    labels={'PCA1': 'Componente principal 1', 'PCA2': 'Componente principal 2', 'PCA3': 'Componente principal 3'}
)

fig.show()
