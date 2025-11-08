import pyprind
import pandas as pd
import numpy as np
import os

# Path to the dataset folder
basepath = 'aclImdb'

# Map folder names ('pos', 'neg') to labels (1, 0)
labels = {'pos': 1, 'neg': 0}

# Initialize a progress bar
pbar = pyprind.ProgBar(50000)

# Initialize a list to store rows
rows = []

# Loop through 'train' and 'test' folders
for s in ('train', 'test'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            rows.append({'review': txt, 'sentiment': labels[l]})
            pbar.update()

# Create the DataFrame
df = pd.DataFrame(rows, columns=['review', 'sentiment'])
print(f"Data loaded. Total reviews: {len(df)}")

# --- NEW CODE: Shuffle and save the DataFrame ---
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))  # Shuffle rows
df.to_csv('movie_data.csv', index=False, encoding='utf-8')  # Save to CSV

# Load from CSV (optional, for verification)
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# Fix column names if needed
try:
    df = df.rename(columns={"0": "review", "1": "sentiment"})
except KeyError:
    pass  # Columns are already correctly named

# Verify
print("\nShuffled DataFrame:")
print(df.head(3))
