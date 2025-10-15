"""
Build FAISS ANN indexes for fast top-K search (no DB required).

Reads:
- data/working/embeddings.parquet  (columns: product_id, img_vec, [txt_vec])

Writes:
- data/working/faiss_img_ivfpq.bin   (may be a Flat index for small N)
- data/working/faiss_txt_ivfpq.bin   (optional if txt_vec present; may be Flat)
- data/working/index_map.parquet     (row order -> product_id mapping)
"""
from typing import Optional
import numpy as np
import pandas as pd
from rich import print
import faiss  # pip install faiss-cpu
from utils import data_path

SMALL_N_CUTOFF = 1000  # below this, use a Flat index (no training)

def maybe_stack(col: pd.Series) -> Optional[np.ndarray]:
    if col is None or col.empty:
        return None
    if col.iloc[0] is None:
        return None
    try:
        X = np.vstack(col.to_list()).astype("float32", copy=False)
        return X
    except Exception:
        return None

def build_index_auto(X: np.ndarray, label: str) -> faiss.Index:
    """
    Choose a sensible FAISS index based on dataset size.
    - If N < SMALL_N_CUTOFF: IndexFlatIP (exact search, no training)
    - Else: IndexIVFPQ with safe nlist and training set size
    Assumes X are L2-normalised; we use IP (cosine similarity).
    """
    assert X.dtype == np.float32
    n, d = X.shape
    if n < SMALL_N_CUTOFF:
        print(f"[blue]{label}[/blue]: N={n} < {SMALL_N_CUTOFF} → using [bold]IndexFlatIP[/bold] (exact search)")
        return faiss.IndexFlatIP(d)

    # Large dataset → IVF-PQ
    # Choose nlist safely: ≤ N, scaled with sqrt(N)
    nlist = min(max(64, int(np.sqrt(n)) * 2), n)
    # Pick m (number of sub-quantisers) that divides d
    for m in (64, 32, 16, 8, 4):
        if d % m == 0:
            break
    else:
        m = 8

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 bits per sub-vector

    # Training set must be >= nlist
    train_n = min(n, max(nlist, 200_000))
    sel = np.arange(train_n) if train_n == n else np.random.default_rng(0).choice(n, size=train_n, replace=False)

    print(f"[blue]{label}[/blue]: N={n} → using [bold]IVF-PQ[/bold] with nlist={nlist}, m={m}, train_n={len(sel)}")
    index.train(X[sel])
    index.add(X)
    return index

def main():
    emb_fp = data_path("working", "embeddings.parquet")
    assert emb_fp.exists(), f"Missing {emb_fp}. Run: python src/embed.py"

    E = pd.read_parquet(emb_fp)
    assert "product_id" in E.columns, "embeddings.parquet must contain product_id"

    # Save row order -> product_id mapping (used by the UI to map FAISS results)
    index_map = E[["product_id"]].copy()
    map_fp = data_path("working", "index_map.parquet")
    index_map.to_parquet(map_fp, index=False)
    print(f"[green]Wrote[/green] {map_fp} (row order mapping for lookups)")

    # ---- Image index ----
    Ximg = maybe_stack(E["img_vec"])
    assert Ximg is not None, "No img_vec found in embeddings.parquet"
    print(f"Image matrix: {Ximg.shape[0]} x {Ximg.shape[1]} (assumed L2-normalised)")

    idx_img = build_index_auto(Ximg, label="Image")
    faiss.write_index(idx_img, str(data_path("working", "faiss_img_ivfpq.bin")))
    print(f"[green]Wrote[/green] faiss_img_ivfpq.bin")

    # ---- Text index (optional) ----
    Xtxt = maybe_stack(E.get("txt_vec"))
    if Xtxt is not None:
        print(f"Text matrix:  {Xtxt.shape[0]} x {Xtxt.shape[1]} (assumed L2-normalised)")
        idx_txt = build_index_auto(Xtxt, label="Text")
        faiss.write_index(idx_txt, str(data_path("working", "faiss_txt_ivfpq.bin")))
        print(f"[green]Wrote[/green] faiss_txt_ivfpq.bin")
    else:
        print("[yellow]No txt_vec column found; skipping text index.[/yellow]")

    print(f"[bold green]DONE[/bold green] — indexes ready.")

if __name__ == "__main__":
    main()
