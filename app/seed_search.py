# app/seed_search.py
# Upload misclassified image → find look-alikes → (top) Send shortlist to Review
# Polished UI + sticky top action bar + FAISS (with fallback) + auto-relaxed threshold

import os, sys, io, math
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from utils import data_path, load_config

# FAISS optional
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# ---------- Page & Style ----------
st.set_page_config(page_title="Find look-alikes from a misclassified image", layout="wide")

st.markdown("""
<style>
:root {
  --brand:#4f46e5; --brand-600:#4f46e5; --brand-100:#eef2ff;
  --ink:#0b0b0c; --muted:#57606a; --panel:#f7f8fb; --stroke:#e8edf3;
}
.block-container{padding-top:1.0rem;padding-bottom:2.0rem;}
.hero { background: linear-gradient(135deg, var(--brand-100), #ffffff);
  border:1px solid var(--stroke); border-radius:18px; padding:18px 20px; }
.kicker {font-size:12px;font-weight:700;letter-spacing:.06em;color:var(--brand-600);text-transform:uppercase;}
.h1 {font-size:22px;font-weight:800;margin:4px 0 6px 0;}
.sub {color:var(--muted);font-size:14px;}
.badge {display:inline-block;padding:2px 8px;border-radius:999px;background:#eaf1ff;color:#0f62fe;font-size:12px;font-weight:600;}
.card {background:var(--panel); border:1px solid var(--stroke); border-radius:16px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.03);}
.topbar {
  position:sticky; top:0; z-index:10; background:#ffffffee; backdrop-filter:blur(6px);
  border-bottom:1px solid var(--stroke); padding:8px 0 10px 0; margin-bottom:8px;
}
.stButton>button {border-radius:10px; padding:.55rem 1rem;}
.stSlider>div>div>div>div[role='slider']{border:2px solid var(--brand);}
.stImage img {border-radius:12px;}
.small {color:var(--muted); font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------- Guards ----------
if not data_path("working", "cluster.parquet").exists() or not data_path("working", "embeddings.parquet").exists():
    st.title("Find look-alikes from a misclassified image")
    st.info("First prepare data and embeddings:")
    st.code("python src/ingest.py\npython src/embed.py\npython src/build_index.py")
    st.stop()

@st.cache_data(show_spinner=False)
def load_embeddings_and_meta():
    emb  = pd.read_parquet(data_path("working","embeddings.parquet"))
    meta = pd.read_parquet(data_path("working","cluster.parquet"))
    return emb.merge(meta, on="product_id", how="left")

df  = load_embeddings_and_meta()
cfg = load_config()

# ---------- Helper fns ----------
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
    st.write("No image")

def _as_int_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    except Exception:
        return s

# Session vars for top-bar action
if "shortlist_ready" not in st.session_state: st.session_state.shortlist_ready = False
if "shortlist_df"   not in st.session_state: st.session_state.shortlist_df   = None
if "df_show"        not in st.session_state: st.session_state.df_show        = None
if "thresh"         not in st.session_state: st.session_state.thresh         = None

# ---------- Sticky top action bar ----------
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.1, 3, 2])
    with c1:
        st.markdown("**Review queue:** " + (f"{len(st.session_state.shortlist_df)} items" if st.session_state.shortlist_ready and st.session_state.shortlist_df is not None else "—"))
        st.caption("The shortlist updates when you upload an image.")
    with c2:
        pass
    with c3:
        top_send = st.button(
            "Send shortlist to Review",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.shortlist_ready
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Hero explainer ----------
st.markdown("""
<div class="hero">
  <div class="kicker">What this tool does</div>
  <div class="h1">Upload one misclassified product → we find others like it</div>
  <div class="sub">
    Drop in a photo of the item that got the wrong insurance. We scan your catalogue for look-alikes,
    create a shortlist for quick review, and generate correction files. No database needed — it runs on your files.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Steps")
    st.markdown("1) **Upload** a misclassified product image\n2) **See similar** products\n3) **Send shortlist** → Review\n4) **Confirm/Reject** → **Export fixes**")
    st.divider()
    st.caption("Scope")
    scope = (cfg.get("scope") or "cluster").lower()
    st.write(f"**Mode:** {'All products' if scope=='all' else 'Insurance cluster'}")
    st.write(f"**Items:** {len(df)}")
    st.divider()
    st.caption("Navigation")
    try:
        st.page_link("app/seed_search.py", label="📤 Upload seed")
        st.page_link("app/pages/review_ui.py", label="✅ Review UI")
    except Exception:
        st.caption("Use the sidebar Pages menu to switch.")

# ---------- Controls & inputs ----------
st.subheader("1) Upload your misclassified product")
left, right = st.columns([1.1, 1.4], gap="large")

with left:
    uploaded = st.file_uploader("Upload the image (PNG/JPG)", type=["png","jpg","jpeg"])
    top_k    = st.slider("How many matches to show", 12, 300, 48, 12)
    sim_min  = st.slider("Initial shortlist threshold", 0.0, 1.0, 0.40, 0.01)
    st.caption("The app will auto-relax this if too few items pass.")

with right:
    if uploaded:
        seed_img = Image.open(io.BytesIO(uploaded.read()))
        st.image(seed_img, caption="Seed image", width="stretch")
    else:
        st.info("Tip: start with a clear product image (front view if available).")

# ---------- Main search ----------
if uploaded:
    # Seed embedding
    seed_vec = embed_image_pil(seed_img).reshape(1,-1).astype("float32")

    # Prefer FAISS if present
    faiss_idx_fp = data_path("working", "faiss_img_ivfpq.bin")
    map_fp       = data_path("working", "index_map.parquet")

    if HAVE_FAISS and faiss_idx_fp.exists() and map_fp.exists():
        index = faiss.read_index(str(faiss_idx_fp))
        SEARCH_K = min(len(df), max(top_k * 5, 200))
        D, I = index.search(seed_vec, SEARCH_K)

        cand = pd.DataFrame({"row": I[0], "sim_img": D[0]})
        cand = cand[cand["row"] >= 0]
        idmap = pd.read_parquet(map_fp).reset_index(drop=True)
        idmap = idmap.reset_index().rename(columns={"index":"row"})
        cand  = cand.merge(idmap[["row","product_id"]], on="row", how="left").dropna(subset=["product_id"])

        cand["product_id"] = _as_int_series(cand["product_id"]).astype("Int64")
        df = df.copy()
        df["product_id"] = _as_int_series(df["product_id"]).astype("Int64")

        cand = cand.groupby("product_id", as_index=False)["sim_img"].max()
        df_show = cand.merge(df, on="product_id", how="left")
        if df_show.empty:
            st.warning("FAISS returned no joinable candidates (likely stale index). Falling back to exact search.")
            img_mat = np.asarray(df["img_vec"].to_list(), dtype="float32")
            sims = (img_mat @ seed_vec.T).ravel()
            df_show = df.copy(); df_show["sim_img"] = sims
        st.caption(f"FAISS: candidates={len(df_show)}")
    else:
        if not HAVE_FAISS:
            st.info("FAISS not installed; using exact search.")
        elif not faiss_idx_fp.exists():
            st.info("FAISS index not found; using exact search.")
        img_mat = np.asarray(df["img_vec"].to_list(), dtype="float32")
        sims = (img_mat @ seed_vec.T).ravel()
        df_show = df.copy(); df_show["sim_img"] = sims

    # ---------- Rank + auto-relax threshold ----------
    df_show = df_show.sort_values("sim_img", ascending=False)
    df_top  = df_show.head(top_k).copy()

    thresh = float(sim_min)
    above  = int((df_show["sim_img"] >= thresh).sum())
    min_needed = min(12, top_k // 2)
    if above == 0:
        thresh = max(0.0, float(df_show["sim_img"].max()) - 1e-6)
    elif above < min_needed:
        keep = min(min_needed, len(df_show))
        if keep > 0:
            thresh = float(df_show["sim_img"].iloc[keep-1])

    df_shortlist = df_show[df_show["sim_img"] >= thresh].copy()

    # Store in session for the top bar button
    st.session_state.shortlist_ready = True
    st.session_state.shortlist_df    = df_shortlist[["product_id"]].copy()
    st.session_state.df_show         = df_show
    st.session_state.thresh          = thresh

    st.toast(f"Candidates: {len(df_show)} · threshold: {thresh:.2f} · shortlisted: {len(df_shortlist)}", icon="🔎")
    st.caption(f"Similarity range: min={df_show['sim_img'].min():.3f} · max={df_show['sim_img'].max():.3f}")

    # ---------- 2) See similar products ----------
    st.subheader("2) Similar products")
    ncols = 3
    rows  = math.ceil(len(df_top) / ncols)
    it    = df_top.iterrows()
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
                st.markdown(f"<div class='small'>PID: {int(r['product_id']) if pd.notna(r['product_id']) else ''} · {r.get('brand','')} · {r.get('country','')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small'>Taxonomy: {r.get('taxonomy_path','')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small'>Insurance: {r.get('insurance_product_name','')}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 3) (Top) Send shortlist to Review ----------
    if top_send:
        queue_path = data_path("output", "review_queue.csv")
        audit_path = data_path("output", "audit_log.csv")
        os.makedirs(queue_path.parent, exist_ok=True)

        out_cols = [
            "product_id","title","taxonomy_path","insurance_product_name",
            "sim_img","image_url","brand","price","country"
        ]
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
