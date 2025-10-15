"""
Create a tiny demo dataset with a seed case and look-alikes.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import random

ROOT = Path(__file__).resolve().parents[1]
IMGS = ROOT / "data" / "input" / "demo_images"
IMGS.mkdir(parents=True, exist_ok=True)

def img(path, label, colour=(180, 180, 180)):
    im = Image.new("RGB", (512, 384), colour)
    d = ImageDraw.Draw(im)
    d.rectangle([30,30,482,354], outline=(0,0,0), width=4)
    d.text((50, 170), label, fill=(20,20,20))
    im.save(path)

rows = []
pid = 1000

# Cameras (correct), and some drones mislabelled as Cameras
for i in range(8):
    pid += 1
    fn = IMGS / f"camera_{i}.jpg"
    img(fn, "CAMERA")
    rows.append({
        "product_id": pid, "title": f"Acme DSLR Camera {i}", "description": "High-res camera",
        "image_url": str(fn), "taxonomy_path": "Electronics > Cameras > DSLRs",
        "brand": "Acme", "price": 499.0+i, "currency": "GBP", "country": "UK"
    })

# Drones mislabelled as Cameras (seed-like)
seed_id = pid + 1
for i in range(6):
    pid += 1
    fn = IMGS / f"drone_{i}.jpg"
    img(fn, "DRONE", colour=(160,200,220))
    title = "FPV quadcopter drone with propeller kit" if i==0 else f"Racer FPV drone {i} with gimbal"
    rows.append({
        "product_id": pid, "title": title, "description": "Brushless motors, FPV, propeller",
        "image_url": str(fn), "taxonomy_path": "Electronics > Cameras > Action Cameras",
        "brand": "SkyCo", "price": 299.0+i, "currency": "GBP", "country": "UK"
    })

# Smartwatches mislabelled as Phones
for i in range(5):
    pid += 1
    fn = IMGS / f"watch_{i}.jpg"
    img(fn, "WATCH", colour=(220,200,160))
    rows.append({
        "product_id": pid, "title": f"Smartwatch with heart rate and GPS {i}", "description": "AMOLED wrist wearable",
        "image_url": str(fn), "taxonomy_path": "Electronics > Mobile Phones",
        "brand": "TimeTech", "price": 159.0+i, "currency": "GBP", "country": "UK"
    })

products = pd.DataFrame(rows)
links = pd.DataFrame({
    "product_id": products["product_id"],
    "insurance_product_id": ["CAMERA_INS_TIER_2"] * len(products),
    "insurance_product_name": ["Camera Insurance – Tier 2"] * len(products)
})

products.to_csv(ROOT / "data" / "input" / "products.csv", index=False)
links.to_csv(ROOT / "data" / "input" / "insurance_links.csv", index=False)

print("Created demo data in data/input/")
print("Seed example product_id (drone mislabelled as Camera):", seed_id)
