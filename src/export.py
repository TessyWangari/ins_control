"""
Export corrections from review decisions.

Reads:
- data/output/review_queue.csv  (from seed_search.py or score.py)
- data/output/audit_log.csv     (from review_ui.py)

Writes:
- data/output/corrections.csv
- data/output/unlink_relink.csv
"""
import pandas as pd
from utils import data_path, ensure_dirs
from datetime import datetime
from rich import print

def guess_proposed_taxonomy(current: str, reasons: str) -> str:
    current = current or ""
    r = (reasons or "").lower()

    # Very simple heuristics; adjust to your catalogue wording
    if "camera" in current and ("drone" in r or "fpv" in r or "propeller" in r):
        return current.replace("Cameras", "Drones").replace("Camera", "Drones")
    if ("mobile phone" in current.lower() or "phone" in current.lower()) and ("watch" in r or "strap" in r):
        return current.replace("Mobile Phones", "Wearables").replace("Phones", "Wearables")
    # fallback: unchanged
    return current

def main():
    ensure_dirs()

    queue_fp = data_path("output", "review_queue.csv")
    audit_fp = data_path("output", "audit_log.csv")

    if not queue_fp.exists():
        raise SystemExit("No review_queue.csv found. Run the upload page and click 'Send shortlist to Review' first.")
    if not audit_fp.exists():
        raise SystemExit("No audit_log.csv found. Open the Review UI and record some decisions first.")

    df_q = pd.read_csv(queue_fp)
    df_a = pd.read_csv(audit_fp)

    # Normalise expected columns (seed_search may not provide these)
    for col, default in [
        ("taxonomy_path", ""),
        ("insurance_product_name", ""),
        ("reasons", ""),        # score.py provides this; seed_search does not
        ("sim_img", 0.0),
        ("sim_txt", 0.0),
        ("shortlist", False),
    ]:
        if col not in df_q.columns:
            df_q[col] = default

    # Merge decisions with queue
    df = df_a.merge(df_q, on="product_id", how="left")

    # Keep only confirmed items for corrections
    df_c = df[df["decision"].str.lower() == "confirm"].copy()
    if df_c.empty:
        print("[yellow]No Confirm decisions found. Nothing to export.[/yellow]")
        return

    # Build a fallback reason if missing
    def build_reason(row):
        if isinstance(row.get("reasons", ""), str) and row["reasons"].strip():
            return row["reasons"]
        # fallback using similarity and shortlist flag
        sim = row.get("sim_img", 0.0)
        sl = row.get("shortlist", False)
        return f"confirmed via review; image similarity={sim:.2f}" + ("; seed shortlist" if sl else "")

    df_c["reason_final"] = [build_reason(r) for _, r in df_c.iterrows()]

    # ---- corrections.csv ----
    df_c["proposed_taxonomy_path"] = [
        guess_proposed_taxonomy(t, r) for t, r in zip(df_c["taxonomy_path"], df_c["reason_final"])
    ]

    corrections = df_c.rename(columns={
        "taxonomy_path": "current_taxonomy_path"
    })[["product_id", "current_taxonomy_path", "proposed_taxonomy_path", "reason_final"]]

    corrections = corrections.assign(
        action="recategorise",
        reviewer=df_c.get("reviewer", "reviewer"),
        reviewed_at=datetime.utcnow().isoformat()
    )
    # Reorder columns nicely
    corrections = corrections[[
        "product_id", "current_taxonomy_path", "proposed_taxonomy_path",
        "action", "reason_final", "reviewer", "reviewed_at"
    ]].rename(columns={"reason_final": "reason"})

    corrections_fp = data_path("output", "corrections.csv")
    corrections.to_csv(corrections_fp, index=False)

    # ---- unlink_relink.csv ----
    curr_ins = df_c.get("insurance_product_name", pd.Series([""] * len(df_c)))
    def propose_insurance(name: str) -> str:
        n = (name or "")
        if "Camera Insurance" in n:
            return n.replace("Camera Insurance", "Drone Insurance")
        if "Phone Insurance" in n or "Mobile Phone Insurance" in n:
            return n.replace("Phone Insurance", "Wearables Insurance").replace("Mobile Phone Insurance", "Wearables Insurance")
        # fallback: unchanged (you may want to leave blank instead)
        return n

    df_c["proposed_insurance"] = [propose_insurance(x) for x in curr_ins]

    unlink = df_c.rename(columns={
        "insurance_product_name": "current_insurance"
    })[["product_id", "current_insurance", "proposed_insurance", "reason_final"]]
    unlink = unlink.rename(columns={"reason_final": "reason"})

    unlink_fp = data_path("output", "unlink_relink.csv")
    unlink.to_csv(unlink_fp, index=False)

    print(f"[green]Wrote[/green] {corrections_fp.name} and {unlink_fp.name} in {corrections_fp.parent}")

if __name__ == "__main__":
    main()
