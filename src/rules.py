\
"""
Keyword/regex rules for simple consistency checks between text and category.
"""
import re
from dataclasses import dataclass
from typing import List, Tuple

DRONE_TERMS = re.compile(r"(quadcopter|propeller|fpv|drone|gimbal|rc)", re.I)
CAMERA_TERMS = re.compile(r"(camera|dslr|mirrorless|lens|point\-and\-shoot)", re.I)

WATCH_TERMS  = re.compile(r"(watch|strap|wrist|heart\s*rate|amoled|ecg|gps)", re.I)
PHONE_TERMS  = re.compile(r"(phone|smartphone|sim|handset|imei)", re.I)

@dataclass
class CheckResult:
    conflict_score: float
    reasons: List[str]

def check_text_vs_category(title: str, taxonomy: str) -> CheckResult:
    title = title or ""
    taxonomy = taxonomy or ""

    reasons = []

    # Drone vs Camera
    has_drone = bool(DRONE_TERMS.search(title))
    has_camera = bool(CAMERA_TERMS.search(title))
    in_camera_cat = "camera" in taxonomy.lower()
    in_drone_cat  = "drone"  in taxonomy.lower()

    # Watch vs Phone
    has_watch = bool(WATCH_TERMS.search(title))
    has_phone = bool(PHONE_TERMS.search(title))
    in_phone_cat = "phone" in taxonomy.lower()
    in_watch_cat = "watch" in taxonomy.lower() or "wearable" in taxonomy.lower()

    conflict = 0.0
    if has_drone and in_camera_cat and not in_drone_cat:
        conflict += 0.6; reasons.append("Title suggests drone, category is Camera")
    if has_watch and in_phone_cat and not in_watch_cat:
        conflict += 0.6; reasons.append("Title suggests watch, category is Phone")
    if has_camera and in_drone_cat:
        conflict += 0.3; reasons.append("Title suggests camera, category is Drone")
    if has_phone and in_watch_cat:
        conflict += 0.3; reasons.append("Title suggests phone, category is Watch")

    return CheckResult(conflict_score=min(conflict, 1.0), reasons=reasons)
