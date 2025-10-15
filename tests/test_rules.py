from rules import check_text_vs_category

def test_drone_in_camera_conflict():
    r = check_text_vs_category("FPV racing drone with propeller kit", "Electronics > Cameras > Action Cameras")
    assert r.conflict_score > 0
    assert any("drone" in s.lower() for s in r.reasons)

def test_watch_in_phone_conflict():
    r = check_text_vs_category("Smartwatch with heart rate and GPS", "Electronics > Mobile Phones")
    assert r.conflict_score > 0
