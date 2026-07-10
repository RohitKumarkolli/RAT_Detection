"""
apk_extractor.py
================================================================
Extracts the 241 binary features used by the TUANDROMD models
from a raw .apk file.

Features:
  1. Android permissions   from AndroidManifest.xml  (212 features)
  2. Dangerous API calls   from DEX bytecode          (28 features)
  3. activityCalled flag   from manifest               (1 feature)

CLI helper:
  python apk_extractor.py TUANDROMD.csv   # regenerates feature_list.json
================================================================
"""

import zipfile, re, json, os, sys
from pathlib import Path

# ── Load feature list ─────────────────────────────────────
_HERE      = Path(__file__).parent
_JSON_PATH = _HERE / "models" / "feature_list.json"


def _load_feature_list() -> list:
    """Load with automatic fallback to CSV generation."""
    # Tier-1: read JSON
    if _JSON_PATH.exists():
        try:
            content = _JSON_PATH.read_text(encoding="utf-8").strip()
            if content:
                fl = json.loads(content)
                if isinstance(fl, list) and fl:
                    print(f"[apk_extractor] Loaded {len(fl)} features from JSON.")
                    return fl
        except Exception as e:
            print(f"[apk_extractor] JSON parse error: {e}")

    # Tier-2: auto-generate from CSV
    csv_candidates = [
        _HERE / "TUANDROMD.csv",
        _HERE.parent / "TUANDROMD.csv",
        Path("TUANDROMD.csv"),
    ]
    for csv in csv_candidates:
        if csv.exists():
            try:
                import pandas as pd
                cols = [c for c in pd.read_csv(csv, nrows=0).columns if c != "Label"]
                if cols:
                    _JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _JSON_PATH.write_text(json.dumps(cols, indent=2), encoding="utf-8")
                    print(f"[apk_extractor] Generated {len(cols)} features from {csv}.")
                    return cols
            except Exception as e:
                print(f"[apk_extractor] CSV error: {e}")

    raise RuntimeError(
        f"Cannot load feature list. Expected: {_JSON_PATH}\n"
        "Run: python apk_extractor.py TUANDROMD.csv")


def generate_feature_list(csv_path: str, save: bool = True) -> list:
    """Public helper – (re)generate feature_list.json from a CSV."""
    import pandas as pd
    cols = [c for c in pd.read_csv(csv_path, nrows=0).columns if c != "Label"]
    if not cols:
        raise ValueError(f"No feature columns found in {csv_path}")
    if save:
        _JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        _JSON_PATH.write_text(json.dumps(cols, indent=2), encoding="utf-8")
        print(f"[apk_extractor] Saved {len(cols)} features → {_JSON_PATH}")
    return cols


FEATURE_LIST        = _load_feature_list()
PERMISSION_FEATURES = [f for f in FEATURE_LIST
                       if not f.startswith("L") and f != "activityCalled"]
API_FEATURES        = [f for f in FEATURE_LIST if f.startswith("L")]

# ── Internal helpers ──────────────────────────────────────

def _decode_binary_xml(data: bytes) -> str:
    """Extract printable ASCII strings from binary AndroidManifest.xml."""
    text = data.decode("latin-1", errors="replace")
    return "\n".join(s for s in re.findall(r"[\x20-\x7e]{4,}", text))


def _extract_permissions(manifest_text: str) -> set:
    found = set()
    for m in re.finditer(
        r"(?:android\.permission\.|android\.Manifest\.permission\.|"
        r"com\.\w+\.permission\.)([A-Z_0-9]+)",
        manifest_text, re.IGNORECASE):
        found.add(m.group(1).upper())
    for perm in PERMISSION_FEATURES:
        if re.search(r"\b" + re.escape(perm) + r"\b", manifest_text, re.IGNORECASE):
            found.add(perm.upper())
    return found


def _extract_dex_strings(apk_path: str) -> str:
    all_strings = []
    with zipfile.ZipFile(apk_path, "r") as z:
        for name in z.namelist():
            if re.match(r"classes\d*\.dex$", name):
                raw = z.read(name)
                all_strings.extend(
                    s.decode("ascii", errors="replace")
                    for s in re.findall(rb"[\x20-\x7e]{5,}", raw))
    return "\n".join(all_strings)


def _has_activity(manifest_text: str) -> bool:
    return bool(re.search(r"\bactivity\b", manifest_text, re.IGNORECASE))

# ── Public API ────────────────────────────────────────────

def extract_features(apk_path: str) -> dict:
    """
    Extract all 241 features from an APK file.
    Returns {'features': {name: 0|1, ...}, 'metadata': {...}}
    """
    features = {f: 0 for f in FEATURE_LIST}
    meta     = {
        "_permissions_found": [],
        "_apis_found"       : [],
        "_apk_size_kb"      : round(os.path.getsize(apk_path) / 1024, 1),
        "_dex_count"        : 0,
        "_error"            : None,
    }
    try:
        with zipfile.ZipFile(apk_path, "r") as z:
            manifest_bytes = z.read("AndroidManifest.xml")
            dex_files      = [n for n in z.namelist() if re.match(r"classes\d*\.dex$", n)]
            meta["_dex_count"] = len(dex_files)

        manifest_text = _decode_binary_xml(manifest_bytes)
        found_perms   = _extract_permissions(manifest_text)

        for perm in PERMISSION_FEATURES:
            if perm.upper() in found_perms:
                features[perm] = 1
                meta["_permissions_found"].append(perm)

        if _has_activity(manifest_text):
            features["activityCalled"] = 1

        dex_text = _extract_dex_strings(apk_path)
        for api in API_FEATURES:
            pattern = re.escape(api).replace(r"\-\>", r"[\->]+")
            if re.search(pattern, dex_text):
                features[api] = 1
                meta["_apis_found"].append(api)

    except zipfile.BadZipFile:
        meta["_error"] = "Not a valid APK/ZIP file."
    except KeyError as e:
        meta["_error"] = f"Missing entry in APK: {e}"
    except Exception as e:
        meta["_error"] = f"Extraction error: {e}"

    return {"features": features, "metadata": meta}


def feature_vector(apk_path: str):
    """
    Returns (vector, metadata).
    vector is a list of 241 floats in FEATURE_LIST order.
    """
    result  = extract_features(apk_path)
    vector  = [float(result["features"][f]) for f in FEATURE_LIST]
    return vector, result["metadata"]


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 2:
        out = generate_feature_list(sys.argv[1], save=True)
        print(f"Done – {len(out)} features saved to {_JSON_PATH}")
    else:
        print(f"Usage: python apk_extractor.py TUANDROMD.csv")
        print(f"Currently loaded: {len(FEATURE_LIST)} features")