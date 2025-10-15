# app/seed_search.py
# Upload misclassified image → find look-alikes → (top) Send shortlist to Review
# Polished UI + sticky top action bar + FAISS (with fallback) + auto-relaxed threshold + bootstrap if data missing

import os, sys, io, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils import data_path, load_config

# Optional FAISS
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# ---------- Page setup ----------
st.set_page_config(page_title="Find look-alikes from a misclassified image", layout="wide")

# ---------- Styles ----------
st.markdown("""
<style>
:root {
  --brand:#4f46e5; --brand-600:#4f46e5; --brand-100:#eef2ff;
  --muted:#57606a; --panel:#f7f8fb; --stroke:#e8edf3;
}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.hero{background:linear-gradient(135deg,var(--brand-100),#ffffff);border:1px solid var(--stroke);
  border-radius:18px;padding:18px 20px;}
.kicker{font-size:12px;font-weight:700;letter-spacing:.06em;color:var(--brand-600);text-transform:uppercase;}
.h1{font-size:22px;font-weight:800;margin:4px 0 6px 0;}
.sub{color:var(--muted);font-size:14px;}
.topbar{position:sticky;top:0;z-index:10;background:#ffffffee;backdrop-filter:blur(6px);
  border-bottom:1px solid var(--stroke);padding:8px 0 10px 0;margin-bottom:8px;}
.stButton>button{border-radius:10px;padding:.55rem 1rem;}
.stSlider>div>div>div>div[role='slider']{border:2px solid var(--brand);}
.stImage img{border-radius:12px;}
.card{background:var(--panel);border:1px solid var(--stroke);border-radius:16px;
  padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.03);}
.small{color:var(--muted);font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def embed_image_pil(pil: Image.Image) -> np.ndarray:
    im = pil.convert("L")
    im = ImageOps.fit(im, (64, 64))
    arr = np.asarray(im, dtype="float32") / 255.0
    v = arr.flatten()
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype("float32")

def display_image(src: str):
    try:
        if isinstance(src, str) and src.lower().startswith(("http://","https://")):
            st.image(src, width="stretch"); return
        p = Path(src)
        if p.exists():
            st.image(str(p), width="stretch"); return
    except Exception:
        pass
    st.caption("No image")

def _as_int_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    except Exception:
        return s

# ---------- Bootstrap (prepare data if missing) ----------
WORKING = data_path("working")
IMGS = data_path("images")
WORKING.mkdir(parents=True, exist_ok=True)

def bootstrap_from_local_images() -> None:
    cl_fp = data_path("working", "cluster.parquet")
    if not cl_fp.exists():
        rows = []
        for p in sorted(IMGS.glob("*")):
            if p.suffix.lower() not in {".jpg",".jpeg",".png"}:
                continue
            pid = p.stem
            rows.append({
                "product_id": int(pid) if pid.isdigit() else pid,
                "title": pid.replace("_"," ").strip(),
                "taxonomy_path": "",
                "insurance_product_name": "",
                "brand": "",
                "price": None,
                "country": "",
                "image_url": str(p).replace("\\","/"),
            })
        if not rows:
            raise RuntimeError("No images found in data/images/.")
        pd.DataFrame(rows).to_parquet(cl_fp, index=False)

    df_cluster = pd.read_parquet(cl_fp)
    embs = []
    for _, r in df_cluster.iterrows():
        src = r.get("image_url","")
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            img = Image.new("RGB", (64,64), (240,240,240))
        v = embed_image_pil(img)
        embs.append({"product_id": r["product_id"], "img_vec": v})
    df_emb = pd.DataFrame(embs)
    df_emb.to_parquet(data_path("working","embeddings.parquet"), index=False)
    df_emb[["product_id"]].to_parquet(data_path("working","index_map.parquet"), index=False)

    # Optional FAISS
    try:
        import faiss
        X = np.vstack(df_emb["img_vec"].to_list()).astype("float32")
        idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X)
        faiss.write_index(idx, str(data_path("working","faiss_img_ivfpq.bin")))
    except Exception:
        pass

def working_ready() -> bool:
    return all((WORKING / n).exists() for n in ["cluster.parquet", "embeddings.parquet"])

if not working_ready():
    st.title("First-time setup")
    st.info("Click **Prepare data now** to build minimal files from your images in `data/images/`.")
    if st.button("Prepare data now", type="primary"):
        with st.status("Preparing data...", expanded=True) as s:
            try:
                bootstrap_from_local_images()
                s.update(label="Done", state="complete")
                st.success("Data prepared. Reloading app...")
                st.experimental_rerun()
            except Exception as e:
                s.update(label="Failed", state="error")
                st.exception(e)
                st.stop()
    st.stop()

# ---------- Load data ----------
@st.cache_data(show_spinner=False)
def load_embeddings_and_meta():
    emb  = pd.read_parquet(data_path("working","embeddings.parquet"))
    meta = pd.read_parquet(data_path("working","cluster.parquet"))
    return emb.merge(meta, on="product_id", how="left")

df = load_embeddings_and_meta()
cfg = load_config()

# ---------- Session vars ----------
if "shortlist_ready" not in st.session_state: st.session_state.shortlist_ready = False
if "shortlist_df"   not in st.session_state: st.session_state.shortlist_df   = None
if "df_show"        not in st.session_state: st.session_state.df_show        = None
if "thresh"         not in st.session_state: st.session_state.thresh         = None

# ---------- Sticky top bar ----------
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        st.markdown("**Review queue:** " + (f"{len(st.session_state.shortlist_df)} items" if st.session_state.shortlist_ready else "—"))
        st.caption("The shortlist updates when you upload an image.")
    with c2: pass
    with c3:
        top_send = st.button("Send shortlist to Review", type="primary", use_container_width=True,
                             disabled=not st.session_state.shortlist_ready)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Hero explainer ----------
st.markdown("""
<div class="hero">
  <div class="kicker">What this tool does</div>
  <div class="h1">Upload one misclassified product → we find others like it</div>
  <div class="sub">
    Upload a photo of the item that got the wrong insurance. We’ll scan your catalogue for similar products,
    build a shortlist for review, and let you generate correction files. No database needed — it runs entirely on your files.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Upload + controls ----------
st.subheader("1) Upload a misclassified product image")
left, right = st.columns([1.1, 1.4], gap="large")

with left:
    uploaded = st.file_uploader("Upload the image (PNG/JPG)", type=["png","jpg","jpeg"])
    top_k = st.slider("How many matches to show", 12, 300, 48, 12)
    sim_min = st.slider("Initial shortlist threshold", 0.0, 1.0, 0.40, 0.01)
    st.caption("The app will auto-relax this if too few items pass.")

with right:
    if uploaded:
        seed_img = Image.open(io.BytesIO(uploaded.read()))
        st.image(seed_img, caption="Seed image", width="stretch")
    else:
        st.info("Tip: start with a clear image of the product you’re testing.")

# ---------- Main logic ----------
if uploaded:
    seed_vec = embed_image_pil(seed_img).reshape(1, -1).astype("float32")

    faiss_idx_fp = data_path("working","faiss_img_ivfpq.bin")
    map_fp = data_path("working","index_map.parquet")

    if HAVE_FAISS and faiss_idx_fp.exists() and map_fp.exists():
        index = faiss.read_index(str(faiss_idx_fp))
        SEARCH_K = min(len(df), max(top_k * 5, 200))
        D, I = index.search(seed_vec, SEARCH_K)
        cand = pd.DataFrame({"row": I[0], "sim_img": D[0]}).query("row >= 0")
        idmap = pd.read_parquet(map_fp).reset_index(drop=True)
        idmap = idmap.reset_index().rename(columns={"index":"row"})
        cand = cand.merge(idmap[["row","product_id"]], on="row", how="left").dropna(subset=["product_id"])
        cand["product_id"] = _as_int_series(cand["product_id"]).astype("Int64")
        df["product_id"] = _as_int_series(df["product_id"]).astype("Int64")
        cand = cand.groupby("product_id", as_index=False)["sim_img"].max()
        df_show = cand.merge(df, on="product_id", how="left")
        if df_show.empty:
            st.warning("FAISS index looks stale. Falling back to exact search.")
            img_mat = np.vstack(df["img_vec"].to_list()).astype("float32")
            sims = (img_mat @ seed_vec.T).ravel()
            df_show = df.copy(); df_show["sim_img"] = sims
    else:
        img_mat = np.vstack(df["img_vec"].to_list()).astype("float32")
        sims = (img_mat @ seed_vec.T).ravel()
        df_show = df.copy(); df_show["sim_img"] = sims

    # ---- Ranking & shortlist ----
    df_show = df_show.sort_values("sim_img", ascending=False)
    df_top = df_show.head(top_k)
    thresh = float(sim_min)
    above = int((df_show["sim_img"] >= thresh).sum())
    min_needed = min(12, top_k // 2)
    if above == 0:
        thresh = max(0.0, float(df_show["sim_img"].max()) - 1e-6)
    elif above < min_needed:
        keep = min(min_needed, len(df_show))
        if keep > 0:
            thresh = float(df_show["sim_img"].iloc[keep-1])
    df_shortlist = df_show[df_show["sim_img"] >= thresh].copy()

    st.session_state.shortlist_ready = True
    st.session_state.shortlist_df = df_shortlist[["product_id"]].copy()
    st.session_state.df_show = df_show
    st.session_state.thresh = thresh

    st.toast(f"Candidates: {len(df_show)} · threshold: {thresh:.2f} · shortlisted: {len(df_shortlist)}", icon="🔎")

    st.subheader("2) Similar products found")
    ncols = 3
    rows = math.ceil(len(df_top) / ncols)
    it = df_top.iterrows()
    for _ in range(rows):
        cols = st.columns(ncols, gap="medium")
        for c in cols:
            try:
                _, r = next(it)
            except StopIteration:
                break
            with c:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                display_image(r.get("image_url",""))
                st.markdown(f"**{r.get('title','(no title)')}**")
                st.markdown(f"<span class='badge'>sim {r['sim_img']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='small'>PID: {r.get('product_id','')} · {r.get('brand','')} · {r.get('country','')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small'>Taxonomy: {r.get('taxonomy_path','')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small'>Insurance: {r.get('insurance_product_name','')}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    if top_send:
        queue_path = data_path("output","review_queue.csv")
        audit_path = data_path("output","audit_log.csv")
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        out_cols = ["product_id","title","taxonomy_path","insurance_product_name","sim_img","image_url","brand","price","country"]
        q_new = st.session_state.df_show[out_cols].copy()
        q_new["shortlist"] = q_new["product_id"].isin(st.session_state.shortlist_df["product_id"])
        q_new.to_csv(queue_path, index=False)

        try:
            a = pd.read_csv(audit_path)
        except FileNotFoundError:
            a = pd.DataFrame(columns=["product_id","decision","reviewer"])
        already = set(a["product_id"].astype(str)) if not a.empty else set()
        add = st.session_state.shortlist_df.copy()
        add = add[~add["product_id"].astype(str).isin(already)]
        if not add.empty:
            add["decision"] = "Unsure (seed shortlist)"
            add["reviewer"] = "reviewer"
            a = pd.concat([a, add], ignore_index=True)
            a.to_csv(audit_path, index=False)

        st.success("Shortlist saved to Review.")
        if hasattr(st, "switch_page"):
            st.switch_page("pages/review_ui.py")
        else:
            st.info("Use the sidebar link to open the Review UI.")
else:
    st.info("Upload the misclassified product image to begin.")
