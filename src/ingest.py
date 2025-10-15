"""
Ingest: build the working dataset according to scope:
- scope=cluster: only products linked to the target insurance_product_id
- scope=all:     ALL products (left-join insurance info if present)

Writes: data/working/cluster.parquet  (name kept for compatibility)
"""
import pandas as pd
from utils import data_path, load_config, ensure_dirs
from rich import print

def main():
    ensure_dirs()
    cfg = load_config()

    inp_products = data_path("input", "products.csv")
    inp_links    = data_path("input", "insurance_links.csv")

    assert inp_products.exists(), f"Missing {inp_products} (provide your export or run scripts/make_demo_data.py)"
    # Links file might not exist; handle gracefully
    has_links = inp_links.exists()

    df_p = pd.read_csv(inp_products)
    df_l = pd.read_csv(inp_links) if has_links else pd.DataFrame(columns=["product_id", "insurance_product_id", "insurance_product_name"])

    # Always LEFT JOIN so unlinked products remain (important for scope=all)
    df = df_p.merge(df_l, on="product_id", how="left")

    scope = (cfg.get("scope") or "cluster").lower()
    if scope not in {"all", "cluster"}:
        print(f"[yellow]Unknown scope '{scope}', defaulting to 'cluster'.[/yellow]")
        scope = "cluster"

    if scope == "cluster":
        cluster_id = cfg.get("insurance_product_id")
        if not cluster_id:
            raise SystemExit("scope=cluster requires 'insurance_product_id' in configs/cluster.yaml")
        before = len(df)
        df = df[df["insurance_product_id"] == cluster_id]
        print(f"[blue]Scope=cluster[/blue]: filtered {before} → {len(df)} rows for insurance_product_id={cluster_id}")
    else:
        print(f"[blue]Scope=all[/blue]: using ALL products ({len(df)} rows)")

    # Optional country filter
    countries = set(cfg.get("countries") or [])
    if countries:
        before = len(df)
        df = df[df["country"].isin(countries)]
        print(f"Applied country filter {countries}: {before} → {len(df)} rows")

    out = data_path("working", "cluster.parquet")
    df.to_parquet(out, index=False)
    print(f"[green]Wrote[/green] {out} with {len(df)} rows")

if __name__ == "__main__":
    main()
