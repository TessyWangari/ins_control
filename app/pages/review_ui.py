# app/pages/review_ui.py
# Review items and generate export files with one click (no printing of Streamlit objects)

import os, sys, math
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd
import streamlit as st
from utils import data_path

st.set_page_config(page_title="Review — Confirm / Reject", layout="wide")

# ---- Styles ----
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
.card { background: var(--secondary-background-color); border: 1px solid #e8edf3; border-radius: 16px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.badge { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#eaf1ff; color:#0f62fe; font-size:12px; font-weight:600; }
.small { color:#57606a; font-size: 13px; }
.stButton>button { border-radius: 10px; }
.stImage img { border-radius: 12px; }
.topbar {
  position:sticky; top:0; z-index:10; background:#ffffffee; backdrop-filter:blur(6px);
  border-bottom:1px solid #e8edf3; padding:8px 0 10px 0; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

st.title("Review shortlisted items")
st.caption("Confirm true issues, reject false alarms, or mark unsure. Then generate correction files with one click.")

queue_path = data_path("output", "review_queue.csv")
audit_path = data_path("output", "audit_log.csv")
out_dir    = data_path("output")

if not queue_path.exists():
    st.warning("No **review_queue.csv** found. Go to the Upload page and click **Send shortlist to Review** first.")
    st.stop()

df = pd.read_csv(queue_path)

# Normalise expected columns for mixed sources
defaults = {
    "sim_img": 0.0, "sim_txt": 0.0, "conflict_score": 0.0,
    "risk_score": None, "bucket": None, "shortlist": False, "reasons": ""
}
for c, v in defaults.items():
    if c not in df.columns:
        df[c] = v

# ---- Sidebar filters ----
with st.sidebar:
    st.header("Filters")
    q = st.text_input("Search title/brand/taxonomy")
    only_sl = df["shortlist"].any() and st.checkbox("Show only shortlisted", value=True)
    sort_by = st.selectbox("Sort by", ["Similarity (image)", "Risk score", "Title A→Z"])
    per_page = st.slider("Cards per page", 9, 60, 18, 3)
    st.divider()
    st.caption("Navigation")
    try:
        st.page_link("app/seed_search.py", label="📤 Upload seed")
        st.page_link("app/pages/review_ui.py", label="✅ Review UI")
    except Exception:
        pass

# Apply filters
mask = pd.Series(True, index=df.index)
if q and q.strip():
    ql = q.lower()
    mask &= (
        df["title"].fillna("").str.lower().str.contains(ql) |
        df["brand"].fillna("").str.lower().str.contains(ql) |
        df["taxonomy_path"].fillna("").str.lower().str.contains(ql)
    )
if only_sl:
    mask &= df["shortlist"] == True
dfv = df[mask].copy()

# Sorting
if sort_by == "Risk score" and dfv["risk_score"].notna().any():
    dfv = dfv.sort_values(["risk_score","sim_img"], ascending=False, na_position="last")
elif sort_by == "Title A→Z":
    dfv = dfv.sort_values("title", ascending=True, na_position="last")
else:
    dfv = dfv.sort_values("sim_img", ascending=False, na_position="last")

st.write(f"Showing **{len(dfv)}** items")

# ---- Helpers (NO returns of Streamlit objects) ----
def render_image(src: str) -> None:
    """Safely render an image without returning/printing Streamlit objects."""
    try:
        if isinstance(src, str) and src.lower().startswith(("http://","https://")):
            st.image(src, width="stretch")
            return
        p = Path(src or "")
        if p.exists():
            st.image(str(p), width="stretch")
            return
    except Exception:
        pass
    st.caption("No image")

def save_decision(pid, decision, reviewer):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        adf = pd.read_csv(audit_path)
    except FileNotFoundError:
        adf = pd.DataFrame(columns=["product_id","decision","reviewer"])
    adf = adf[adf["product_id"].astype(str) != str(pid)]
    adf = pd.concat([adf, pd.DataFrame([{"product_id": pid, "decision": decision, "reviewer": reviewer}])], ignore_index=True)
    adf.to_csv(audit_path, index=False)

# ---- Sticky topbar with Export button ----
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 2])
    with c1:
        st.markdown("**Export:** Create catalogue correction files from your **Confirm** decisions.")
        st.caption("We’ll generate *corrections.csv* and *unlink_relink.csv* and show a preview below.")
    with c2:
        gen = st.button("Generate export files", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Pagination ----
n = len(dfv)
pages = max(1, math.ceil(n / per_page))
page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
start, end = (page-1)*per_page, min(page*per_page, n)
page_df = dfv.iloc[start:end]

# ---- Cards grid ----
cols_per_row = 3
rows = math.ceil(len(page_df) / cols_per_row)
it = page_df.iterrows()

for _ in range(rows):
    cols = st.columns(cols_per_row, gap="medium")
    for c in cols:
        try:
            _, r = next(it)
        except StopIteration:
            break
        with c:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # ---- IMAGE (safe) ----
            src = r.get("image_url","")
            render_image(src)

            # ---- TEXT + ACTIONS ----
            st.markdown(f"**{r.get('title','(no title)')}**")
            st.markdown(f"<span class='badge'>sim {float(r.get('sim_img',0.0)):.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'>PID: {int(r['product_id']) if pd.notna(r['product_id']) else ''} · {r.get('brand','')} · {r.get('country','')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'>Taxonomy: {r.get('taxonomy_path','')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'>Insurance: {r.get('insurance_product_name','')}</div>", unsafe_allow_html=True)
            if r.get("reasons",""):
                st.markdown(f"<div class='small'><b>Reasons:</b> {r.get('reasons','')}</div>", unsafe_allow_html=True)

            reviewer = st.text_input("Reviewer", key=f"rev_{r['product_id']}", value="reviewer")
            b1, b2, b3 = st.columns(3)
            if b1.button("Confirm", key=f"c_{r['product_id']}"):
                save_decision(r["product_id"], "Confirm", reviewer); st.toast("Saved: Confirm ✅", icon="✅")
            if b2.button("Reject",  key=f"r_{r['product_id']}"):
                save_decision(r["product_id"], "Reject", reviewer);  st.toast("Saved: Reject ❌", icon="❌")
            if b3.button("Unsure",  key=f"u_{r['product_id']}"):
                save_decision(r["product_id"], "Unsure", reviewer);  st.toast("Saved: Unsure 🤔", icon="🤔")

            st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---- Export logic (inline, no terminal needed) ----
def run_export_inline():
    try:
        a = pd.read_csv(audit_path)
    except FileNotFoundError:
        a = pd.DataFrame(columns=["product_id","decision","reviewer"])

    if a.empty or "Confirm" not in set(a["decision"]):
        st.warning("No **Confirm** decisions found in audit_log.csv. Mark some items as Confirm first.")
        return

    merged = df.merge(a[["product_id","decision","reviewer"]], on="product_id", how="left")
    confirmed = merged[merged["decision"] == "Confirm"].copy()
    if confirmed.empty:
        st.warning("No confirmed items after merge. Check product_id types.")
        return

    corr_cols = ["product_id","title","taxonomy_path","image_url","brand","price","country"]
    corr = confirmed[corr_cols].copy()
    corr["proposed_taxonomy_path"] = ""   # placeholder for ops / smarter mapper
    corr["note"] = "Confirmed misclassification"

    url_cols = ["product_id","title","insurance_product_name","country"]
    rel = confirmed[url_cols].copy().rename(columns={"insurance_product_name":"current_insurance"})
    rel["unlink"] = rel["current_insurance"].fillna("")
    rel["relink"] = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    corr_fp = out_dir / "corrections.csv"
    rel_fp  = out_dir / "unlink_relink.csv"
    corr.to_csv(corr_fp, index=False)
    rel.to_csv(rel_fp, index=False)

    st.success("Export complete. Files written to data/output/")

    st.subheader("corrections.csv (preview)")
    st.dataframe(corr.head(200))
    st.download_button(
        "Download corrections.csv",
        data=corr.to_csv(index=False).encode("utf-8"),
        file_name="corrections.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.subheader("unlink_relink.csv (preview)")
    st.dataframe(rel.head(200))
    st.download_button(
        "Download unlink_relink.csv",
        data=rel.to_csv(index=False).encode("utf-8"),
        file_name="unlink_relink.csv",
        mime="text/csv",
        use_container_width=True
    )

if gen:
    run_export_inline()

st.caption("Tip: You can still run `python src/export.py` from the terminal if you prefer batch mode.")
