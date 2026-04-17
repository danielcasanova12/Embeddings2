# prepare_bvcc_csv.py
import pandas as pd

base = "/home/time_mos/casanova/CAL-MOS/data/dataset/BVCC/main/DATA/sets"

splits = {"train": "train_mos_list.txt", "val": "val_mos_list.txt", "test": "test_mos_list.txt"}

dfs = []
for split, fname in splits.items():
    df = pd.read_csv(f"{base}/{fname}", header=None, names=["filename", "mos"])
    df["split"] = split
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
full_df.to_csv("bvcc.csv", index=False)
print(full_df.head())
print(f"Total: {len(full_df)} amostras")