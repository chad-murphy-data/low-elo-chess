import pandas as pd

df = pd.read_csv("data/moves_features.csv")
sample = df.sample(n=10000, random_state=42)
sample.to_csv("data/moves_features_sample.csv", index=False)

print(f"Full dataset: {len(df):,} rows, {len(df.columns)} columns")
print(f"Sample: {len(sample):,} rows")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nHead:\n{sample.head()}")