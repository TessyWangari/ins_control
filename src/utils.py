import yaml
from pathlib import Path
from typing import Any, Dict
from rich import print

ROOT = Path(__file__).resolve().parents[1]

def data_path(*p): return ROOT / "data" / Path(*p)
def cfg_path(*p): return ROOT / "configs" / Path(*p)

def load_config() -> Dict[str, Any]:
    with open(cfg_path("cluster.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    for d in ["input", "working", "output"]:
        (ROOT / "data" / d).mkdir(parents=True, exist_ok=True)
