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
df['Aggression'] = (df['Attack'] + df['Sp. Atk']) / df['Total']
df['Endurance']  = (df['HP'] + df['Defense'] + df['Sp. Def']) / df['Total']
df['Mobility']   = df['Speed'] / df['Total']
df['Specialist'] = abs(df['Attack'] - df['Sp. Atk']) / df['Total']
df['Balance']    = df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].std(axis=1, ddof=0)

# Selección de features
features = ['Aggression', 'Endurance', 'Mobility', 'Specialist', 'Balance']
X = df[features]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(df[features].head())  # o X_scaled para ver los datos listos

print("Datos procesados y escalados correctamente.")
# Guardar el DataFrame procesado si es necesario
#print(X_scaled.head())
