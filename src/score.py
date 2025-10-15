"""
Join candidates with metadata, apply risk scoring, and auto-shortlist high-similarity items.
Writes:
- data/output/review_queue.csv
- (optionally) appends to data/output/audit_log.csv with "Unsure (auto-shortlist)"
"""
import pandas as pd
from utils import data_path, load_config, ensure_dirs
from rules import check_text_vs_category
from rich import print

def main():
    ensure_dirs()
    cfg = load_config()
    thr_high = float(cfg["thresholds"]["high"])
    thr_grey = float(cfg["thresholds"]["grey"])

    df_c = pd.read_parquet(data_path("working", "candidates.parquet"))
    df_m = pd.read_parquet(data_path("working", "cluster.parquet"))

    df = df_c.merge(df_m, on="product_id", how="left")

    # Consistency checks
    conflicts = df.apply(lambda r: check_text_vs_category(r["title"], r["taxonomy_path"]), axis=1)
    df["conflict_score"] = [c.conflict_score for c in conflicts]
    df["reasons"] = ["; ".join(c.reasons) for c in conflicts]

    # Risk score (unchanged)
    df["risk_score"] = (
        0.5 * df["sim_img"].fillna(0) +
        0.3 * df["sim_txt"].fillna(0) +
        0.2 * df["conflict_score"].fillna(0)
    )

    def bucket(x):
        if x >= thr_high: return "HIGH"
        if x >= thr_grey: return "GREY"
        return "LOW"
    df["bucket"] = df["risk_score"].apply(bucket)

    # -------- Auto-shortlist logic --------
    sl_cfg = (cfg.get("shortlist") or {})
    sl_enable = bool(sl_cfg.get("enable", False))
    sl_by = (sl_cfg.get("by") or "combined").lower()

    sim_img = df["sim_img"].fillna(0)
    sim_txt = df["sim_txt"].fillna(0)
    sim_combined = 0.5 * sim_img + 0.5 * sim_txt

    if sl_by == "img":
        shortlist_mask = sim_img >= float(sl_cfg.get("sim_img_min", 0.8))
    elif sl_by == "txt":
        shortlist_mask = sim_txt >= float(sl_cfg.get("sim_txt_min", 0.6))
    else:
        shortlist_mask = sim_combined >= float(sl_cfg.get("combined_min", 0.7))

    df["shortlist"] = shortlist_mask

    # Save review queue
    cols = [
        "product_id","title","taxonomy_path","insurance_product_name",
        "sim_img","sim_txt","conflict_score","risk_score","bucket",
        "reasons","image_url","brand","price","country","shortlist"
    ]
    out_queue = data_path("output", "review_queue.csv")
    df[cols].to_csv(out_queue, index=False)
    print(f"[green]Wrote[/green] {out_queue} with {len(df)} rows "
          f"({df['shortlist'].sum()} auto-shortlisted)")

    # Append auto-shortlisted items to audit_log.csv as “Unsure (auto-shortlist)”
    if sl_enable:
        auto_df = df.loc[df["shortlist"], ["product_id"]].copy()
        if not auto_df.empty:
            audit_path = data_path("output", "audit_log.csv")
            try:
                existing = pd.read_csv(audit_path)
            except FileNotFoundError:
                existing = pd.DataFrame(columns=["product_id","decision","reviewer"])

            # avoid duplicates
            already = set(existing["product_id"].astype(int)) if not existing.empty else set()
            to_add = auto_df[~auto_df["product_id"].astype(int).isin(already)].copy()
            to_add["decision"] = "Unsure (auto-shortlist)"
            to_add["reviewer"] = "reviewer"

            if not to_add.empty:
                new_audit = pd.concat([existing, to_add], ignore_index=True)
                new_audit.to_csv(audit_path, index=False)
                print(f"[green]Updated[/green] {audit_path} (+{len(to_add)} auto-shortlisted)")
    # -------- end shortlist --------

if __name__ == "__main__":
    main()
