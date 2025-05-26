import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "abcsds/pokemon",
    path="Pokemon.csv"
)

print(df.head())
