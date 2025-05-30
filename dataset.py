import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "abcsds/pokemon",
    path="Pokemon.csv"
)

required_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']
for col in required_columns:
    assert col in df.columns, f"Falta la columna: {col}"

# Crear rasgos de comportamiento
df['Agresividad'] = (df['Attack'] + df['Sp. Atk']) / df['Total']

df['Resistencia'] = (df['HP'] * 0.5 + df['Defense'] + df['Sp. Def']) / (df['Total'] + df['HP'] * 0.5)

df['Movilidad'] = df['Speed'] / df['Total']

df['Especialista ofensivo'] = abs(df['Attack'] - df['Sp. Atk']) / (df['Attack'] + df['Sp. Atk'])

df['Especialista defensivo'] = abs(df['Defense'] - df['Sp. Def']) / (df['Defense'] + df['Sp. Def'])

df['Balanceado'] = df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].std(axis=1, ddof=0) / \
                   df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].mean(axis=1)



# Selección de features
features = ['Agresividad', 'Resistencia', 'Movilidad', 'Especialista ofensivo', 'Especialista defensivo', 'Balanceado']
X = df[features]

print(df[features].head())  # o X_scaled para ver los datos listos

print("Datos procesados y escalados correctamente.")
# Guardar el DataFrame procesado si es necesario
df.to_csv("Pokemon_processed.csv", index=False)
#print(X_scaled.head())
