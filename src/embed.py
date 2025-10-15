"""
Embeddings step (offline-friendly, sharper image fingerprints)

- TEXT: HashingVectorizer (+ L2 normalisation). No internet required.
- IMAGE: Grayscale downsample (now 64x64) → 4096-D vector.
         Supports local file paths and http(s) URLs with on-disk caching.

Reads:  data/working/cluster.parquet
Writes: data/working/embeddings.parquet
"""

from pathlib import Path
from typing import Optional
import hashlib
import io

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from utils import data_path, ensure_dirs
from rich import print

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Deliberately offline: we do NOT use SentenceTransformers here.
# Remote images allowed; cached under data/working/image_cache
FETCH_REMOTE = True
CACHE_DIR = data_path("working", "image_cache")

try:
    import requests  # type: ignore
except Exception:
    requests = None
    FETCH_REMOTE = False
    print("[yellow]requests not available; remote image fetching disabled.[/yellow]")

# -----------------------------------------------------------------------------
# Text embedding (offline)
# -----------------------------------------------------------------------------
def embed_text(texts):
    """
    Offline text embedding using HashingVectorizer with L2 normalisation.
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize

    vec = HashingVectorizer(
        n_features=512,
        alternate_sign=False,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vec.transform(texts).astype("float32")
    X = normalize(X)
    return X.toarray().astype("float32")

# -----------------------------------------------------------------------------
# Image loading + embedding
# -----------------------------------------------------------------------------
def _cache_key(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _load_image_from_source(src: str) -> Optional[Image.Image]:
    """
    Return a PIL Image from a local path OR http(s) URL (with caching), else None.
    """
    if not src:
        return None

    # Local path
    try:
        p = Path(src)
        if p.exists():
            return Image.open(p).convert("RGB")
    except Exception:
        pass

    # Remote URL
    if FETCH_REMOTE and isinstance(src, str) and src.lower().startswith(("http://", "https://")) and requests:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            ck = _cache_key(src)
            cached = CACHE_DIR / f"{ck}.img"
            if cached.exists():
                return Image.open(cached).convert("RGB")
            r = requests.get(src, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            try:
                im.save(cached)
            except Exception:
                pass
            return im
        except Exception:
            return None

    return None

def embed_image(sources):
    """
    Lightweight image embedding:
    - Convert to greyscale
    - Centre-crop/fit to 64x64
    - Flatten to 4096-D and L2-normalise
    If load fails, returns a zero vector (still L2-normalised safely).
    """
    ok, fail = 0, 0
    vecs = []
    for src in sources:
        try:
            im = _load_image_from_source(str(src) if src is not None else "")
            if im is None:
                fail += 1
                v = np.zeros((64 * 64,), dtype="float32")
            else:
                im = ImageOps.fit(im.convert("L"), (64, 64))
                arr = np.asarray(im, dtype="float32") / 255.0
                v = arr.flatten()
                ok += 1
        except Exception:
            fail += 1
            v = np.zeros((64 * 64,), dtype="float32")

        # L2 normalise (safe even for zeros)
        n = np.linalg.norm(v) + 1e-8
        vecs.append((v / n).astype("float32"))

    print(f"[cyan]Image embedding[/cyan]: ok={ok}, failed={fail}")
    return np.vstack(vecs)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ensure_dirs()

    cluster_fp = data_path("working", "cluster.parquet")
    if not cluster_fp.exists():
        raise SystemExit(
            "Missing working dataset. Run:\n  python src/ingest.py\n  (then re-run this script)"
        )

    df = pd.read_parquet(cluster_fp)

    # Sanity checks
    expected_cols = ["product_id", "title", "description", "image_url"]
    for c in expected_cols:
        if c not in df.columns:
            raise SystemExit(f"Expected column '{c}' not found in {cluster_fp}")

    # Compose text field (title + description)
    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    txt_vecs = embed_text(texts)

    # Image sources (may be URLs or local paths)
    img_sources = df["image_url"].fillna("").tolist()
    img_vecs = embed_image(img_sources)

    # Write output
    out = data_path("working", "embeddings.parquet")
    out_df = pd.DataFrame(
        {
            "product_id": df["product_id"].tolist(),
            "txt_vec": [v.tolist() for v in txt_vecs],
            "img_vec": [v.tolist() for v in img_vecs],
        }
    )
    out_df.to_parquet(out, index=False)
    print(f"[green]Wrote[/green] {out} with {len(out_df)} embeddings")

if __name__ == "__main__":
    main()
