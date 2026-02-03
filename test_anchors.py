from src.attribution.anchor_models import AnchorModelsDatabase

db = AnchorModelsDatabase()
anchors = db.list_all_anchors()
print(f"Total anchors: {len(anchors)}")
for model_name, info in anchors.items():
    has_fp = info.get('has_fingerprint', False)
    # Load fingerprint to get size
    if has_fp:
        fp = db.load_fingerprint(model_name)
        fp_size = len(fp) if fp else 0
    else:
        fp_size = 0
    print(f"  {model_name}: {'✓' if has_fp else '✗'} ({fp_size} features)")
