import os
import csv
import pandas as pd


df = pd.read_csv("/home/hxxzhang/SafeDiff/dataset/i2p_benchmark.csv", usecols=['prompt'])
# Shuffle the DataFrame
df_shuffled = df.sample(frac=0.02).reset_index(drop=True)

df_shuffled['harm'] = 1

df_shuffled.to_csv("/home/hxxzhang/SafeDiff/dataset/test.csv", index=False)