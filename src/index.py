"""
Index & retrieve top-K neighbours within the cluster.
Usage: python src/index.py --seed <product_id>
Writes data/working/candidates.parquet
"""
import argparse
import numpy as np
import pandas as pd
import faiss
from utils import data_path, load_config, ensure_dirs
from rich import print

def np_from_series(series: pd.Series) -> np.ndarray:
    arr = np.asarray(series.to_list(), dtype="float32")
    return np.ascontiguousarray(arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed product_id in the cluster")
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config()

    df_e = pd.read_parquet(data_path("working", "embeddings.parquet"))
    if df_e.empty:
        raise SystemExit("No embeddings found. Did you run src/ingest.py and src/embed.py?")

    # Make sure the seed exists in this cluster
    if args.seed not in set(df_e["product_id"].astype(int)):
        raise SystemExit(f"Seed {args.seed} not found in embeddings for this cluster.")

    # Cap K to dataset size (FAISS can emit -1 labels if asking for more than available with some indexes)
    K_cfg = int(cfg.get("top_k", 200))
    K = min(K_cfg, len(df_e))

    # Build matrices
    txt = np_from_series(df_e["txt_vec"])
    img = np_from_series(df_e["img_vec"])

    # Build flat (exact) IP indexes; vectors are already L2-normalised
    idx_t = faiss.IndexFlatIP(txt.shape[1])
    idx_i = faiss.IndexFlatIP(img.shape[1])
    idx_t.add(txt)
    idx_i.add(img)

    # Locate seed position (positional index)
    seed_positions = df_e.index[df_e["product_id"].astype(int) == args.seed].to_list()
    pos = seed_positions[0]

    D_t, I_t = idx_t.search(txt[pos:pos+1], K)
    D_i, I_i = idx_i.search(img[pos:pos+1], K)

    # Union + de-dupe; guard against invalid indices (e.g., -1)
    sims = {}
    n = len(df_e)

    def add_hits(I, D, key):
        for i, d in zip(I[0], D[0]):
            if i is None or int(i) < 0 or int(i) >= n:
                continue
            pid = int(df_e.iloc[int(i)]["product_id"])
            entry = sims.setdefault(pid, {})
            entry[key] = float(d)

    add_hits(I_t, D_t, "sim_txt")
    add_hits(I_i, D_i, "sim_img")

    rows = [
        {
            "product_id": pid,
            "sim_txt": vals.get("sim_txt", 0.0),
            "sim_img": vals.get("sim_img", 0.0),
        }
        for pid, vals in sims.items()
    ]
    df_c = pd.DataFrame(rows).sort_values(by=["sim_img", "sim_txt"], ascending=False)

    out = data_path("working", "candidates.parquet")
    df_c.to_parquet(out, index=False)
    print(f"[green]Wrote[/green] {out} with {len(df_c)} candidates (K asked={K_cfg}, used={K})")

if __name__ == "__main__":
    main()
