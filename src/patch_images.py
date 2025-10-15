# src/patch_images.py
import pandas as pd
from pathlib import Path

from utils import data_path

df = pd.read_parquet(data_path("working","cluster.parquet"))

# assume your filenames are product_id with common extensions
def guess_path(pid):
    base = data_path("images")
    for ext in (".jpg",".jpeg",".png"):
        p = base / f"{pid}{ext}"
        if p.exists():
            return str(p).replace("\\","/")
    return ""

if "image_url" not in df.columns:
    df["image_url"] = ""

df["image_url"] = [
    x if isinstance(x, str) and x.strip() else guess_path(pid)
    for pid, x in zip(df["product_id"], df.get("image_url",""))
]

df.to_parquet(data_path("working","cluster.parquet"), index=False)
print("Patched cluster.parquet with local image paths.")
