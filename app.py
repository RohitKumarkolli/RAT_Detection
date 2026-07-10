# """
# app.py  -  Flask APK Malware Detection Interface
# ================================================================
# Routes:
#   GET  /                    →  APK upload page
#   POST /analyze             →  analyze uploaded APK  →  JSON
#   GET  /result/<job_id>     →  result page
#   GET  /test                →  manual feature tester
#   POST /api/test-features   →  run models on manual vector  →  JSON
#   GET  /models              →  model info page
#   GET  /api/features        →  feature list JSON
# ================================================================
# """

# import os, uuid, json, time, traceback
# import numpy as np
# from pathlib import Path
# from flask   import Flask, render_template, request, jsonify, redirect, url_for
# from werkzeug.utils import secure_filename
# import joblib

# from apk_extractor import feature_vector, FEATURE_LIST

# # ── Config ────────────────────────────────────────────────
# BASE_DIR      = Path(__file__).parent
# UPLOAD_FOLDER = BASE_DIR / "uploads"
# MODEL_DIR     = BASE_DIR / "models"
# ALLOWED_EXT   = {"apk"}
# MAX_MB        = 100

# UPLOAD_FOLDER.mkdir(exist_ok=True)

# app = Flask(__name__)
# app.secret_key = os.urandom(24)
# app.config["UPLOAD_FOLDER"]      = str(UPLOAD_FOLDER)
# app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# JOBS: dict = {}   # in-memory job store

# # ── Model loader ──────────────────────────────────────────
# MODELS: dict = {}

# def load_models():
#     specs = {
#         "Random Forest": "random_forest/rf_model.pkl",
#         "Decision Tree": "decision_tree/dt_model.pkl",
#         "XGBoost"      : "xgboost/xgb_model.pkl",
#         "AdaBoost"     : "adaboost/adaboost_model.pkl",
#         "Stacking"     : "stacking/stacking_model.pkl",
#     }
#     for name, rel in specs.items():
#         full = MODEL_DIR / rel
#         if full.exists():
#             try:
#                 MODELS[name] = joblib.load(full)
#                 print(f"  ✓  Loaded {name}")
#             except Exception as e:
#                 print(f"  ✗  Failed {name}: {e}")
#         else:
#             print(f"  ⚠  Not found: {full}  (run training script first)")

# load_models()

# # ── Feature categories (for manual tester) ───────────────
# DANGEROUS_PERMS = {
#     "READ_SMS","SEND_SMS","RECEIVE_SMS","RECORD_AUDIO",
#     "READ_CONTACTS","WRITE_CONTACTS","READ_CALL_LOG","WRITE_CALL_LOG",
#     "CAMERA","ACCESS_FINE_LOCATION","ACCESS_COARSE_LOCATION",
#     "CALL_PHONE","READ_PHONE_STATE","PROCESS_OUTGOING_CALLS",
#     "GET_ACCOUNTS","WRITE_EXTERNAL_STORAGE","READ_EXTERNAL_STORAGE",
#     "SYSTEM_ALERT_WINDOW","INSTALL_PACKAGES","DELETE_PACKAGES",
#     "RECEIVE_BOOT_COMPLETED","ACCESS_SUPERUSER","MASTER_CLEAR",
#     "REBOOT","BRICK","MODIFY_PHONE_STATE",
# }

# def _categorise_features():
#     groups = {
#         "Dangerous Permissions": [],
#         "Network & Connectivity": [],
#         "Storage & Files": [],
#         "System & Device": [],
#         "Accounts & Sync": [],
#         "Media & Sensors": [],
#         "Other Permissions": [],
#         "Dangerous API Calls": [],
#     }
#     net  = {"NETWORK","WIFI","INTERNET","BLUETOOTH","NFC","CHANGE_NETWORK","ACCESS_WIFI","ACCESS_NETWORK"}
#     stor = {"STORAGE","EXTERNAL_STORAGE","INTERNAL_STORAGE","SDCARD","MOUNT","CLEAR_APP","DELETE_CACHE"}
#     sys_ = {"SYSTEM","DEVICE","REBOOT","BRICK","MASTER_CLEAR","INJECT","FACTORY","INSTALL_PACKAGES",
#              "DELETE_PACKAGES","CHANGE_COMPONENT","BIND_DEVICE","CHANGE_CONFIG","SET_TIME",
#              "RECEIVE_BOOT","SUPERUSER","MODIFY_PHONE","WAKE_LOCK"}
#     acc  = {"ACCOUNT","SYNC","AUTH","GET_ACCOUNTS","MANAGE_ACCOUNTS","AUTHENTICATE"}
#     med  = {"CAMERA","AUDIO","RECORD","MEDIA","VIBRATE","SENSOR","BODY_SENSOR","FLASHLIGHT","TRANSMIT_IR"}

#     for feat in FEATURE_LIST:
#         if feat.startswith("L"):
#             groups["Dangerous API Calls"].append(feat)
#         elif feat in DANGEROUS_PERMS:
#             groups["Dangerous Permissions"].append(feat)
#         elif any(k in feat for k in net):
#             groups["Network & Connectivity"].append(feat)
#         elif any(k in feat for k in stor):
#             groups["Storage & Files"].append(feat)
#         elif any(k in feat for k in sys_):
#             groups["System & Device"].append(feat)
#         elif any(k in feat for k in acc):
#             groups["Accounts & Sync"].append(feat)
#         elif any(k in feat for k in med):
#             groups["Media & Sensors"].append(feat)
#         else:
#             groups["Other Permissions"].append(feat)
#     return {k: v for k, v in groups.items() if v}

# FEATURE_GROUPS = _categorise_features()

# # ── Core inference helpers ────────────────────────────────

# def allowed_file(fn):
#     return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# def run_models(vector: list) -> dict:
#     X = np.array(vector, dtype=np.float32).reshape(1, -1)
#     results = {}
#     for name, bundle in MODELS.items():
#         try:
#             model = bundle["model"]
#             le    = bundle["label_encoder"]
#             vt    = bundle.get("variance_filter")
#             X_in  = vt.transform(X) if vt is not None else X
#             proba = model.predict_proba(X_in)[0]
#             pred  = le.inverse_transform([int(np.argmax(proba))])[0]
#             classes   = list(le.classes_)
#             mal_idx   = classes.index("malware") if "malware" in classes else 1
#             mal_prob  = float(proba[mal_idx])
#             results[name] = {
#                 "prediction":    pred,
#                 "malware_prob":  round(mal_prob * 100, 2),
#                 "goodware_prob": round((1 - mal_prob) * 100, 2),
#                 "confidence":    round(float(max(proba)) * 100, 2),
#             }
#         except Exception as e:
#             results[name] = {"error": str(e)}
#     return results


# def ensemble_verdict(model_results: dict) -> dict:
#     preds, probs = [], []
#     for r in model_results.values():
#         if "error" not in r:
#             preds.append(r["prediction"])
#             probs.append(r["malware_prob"])
#     if not preds:
#         return {"verdict":"unknown","score":0,"risk_level":"Unknown",
#                 "confidence":0,"mal_votes":0,"good_votes":0,"total_models":0}
#     mal_v  = preds.count("malware")
#     good_v = preds.count("goodware")
#     avg    = round(sum(probs) / len(probs), 2)
#     verdict= "malware" if mal_v >= good_v else "goodware"
#     conf   = round(max(mal_v, good_v) / len(preds) * 100, 1)
#     risk   = ("Critical" if avg >= 80 else "High"   if avg >= 60 else
#               "Medium"   if avg >= 40 else "Low"    if avg >= 20 else "Safe")
#     return {"verdict": verdict, "score": avg, "risk_level": risk,
#             "confidence": conf, "mal_votes": mal_v, "good_votes": good_v,
#             "total_models": len(preds)}

# # ── Routes ────────────────────────────────────────────────

# @app.route("/")
# def index():
#     return render_template("index.html",
#                            model_names=list(MODELS.keys()) or ["No models loaded"],
#                            models_loaded=len(MODELS))


# @app.route("/analyze", methods=["POST"])
# def analyze():
#     if "apk_file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     f = request.files["apk_file"]
#     if not f.filename:
#         return jsonify({"error": "Empty filename"}), 400
#     if not allowed_file(f.filename):
#         return jsonify({"error": "Only .apk files accepted"}), 400
#     if not MODELS:
#         return jsonify({"error": "No models loaded – run training scripts first"}), 503

#     job_id   = str(uuid.uuid4())
#     filename = secure_filename(f.filename)
#     apk_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{filename}")
#     f.save(apk_path)

#     try:
#         t0                   = time.time()
#         vec, meta            = feature_vector(apk_path)
#         extract_ms           = round((time.time() - t0) * 1000)
#         t1                   = time.time()
#         model_results        = run_models(vec)
#         infer_ms             = round((time.time() - t1) * 1000)
#         verdict              = ensemble_verdict(model_results)
#         active               = [FEATURE_LIST[i] for i, v in enumerate(vec) if v == 1.0]
#         permissions_active   = [f for f in active if not f.startswith("L") and f != "activityCalled"]
#         apis_active          = [f for f in active if f.startswith("L")]

#         result = {
#             "job_id": job_id, "filename": filename,
#             "apk_size_kb": meta["_apk_size_kb"], "dex_count": meta["_dex_count"],
#             "extract_error": meta["_error"], "extract_ms": extract_ms, "infer_ms": infer_ms,
#             "features_total": len(FEATURE_LIST), "features_active": len(active),
#             "permissions_active": permissions_active, "apis_active": apis_active,
#             "model_results": model_results, "verdict": verdict,
#         }
#         JOBS[job_id] = result
#         try: os.remove(apk_path)
#         except: pass
#         return jsonify(result)

#     except Exception as e:
#         try: os.remove(apk_path)
#         except: pass
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# @app.route("/result/<job_id>")
# def result_page(job_id):
#     job = JOBS.get(job_id)
#     if not job:
#         return redirect(url_for("index"))
#     return render_template("result.html", result=job)


# @app.route("/test")
# def test_page():
#     return render_template("test.html",
#                            feature_groups=FEATURE_GROUPS,
#                            feature_list=FEATURE_LIST,
#                            dangerous_perms=list(DANGEROUS_PERMS),
#                            models_loaded=len(MODELS),
#                            model_names=list(MODELS.keys()),
#                            total_features=len(FEATURE_LIST))


# @app.route("/api/test-features", methods=["POST"])
# def api_test_features():
#     if not MODELS:
#         return jsonify({"error": "No models loaded"}), 503
#     try:
#         data   = request.get_json(force=True) or {}
#         active = {k: int(bool(v)) for k, v in data.items() if k in set(FEATURE_LIST)}
#         vector = [float(active.get(f, 0)) for f in FEATURE_LIST]
#         t0             = time.time()
#         model_results  = run_models(vector)
#         infer_ms       = round((time.time() - t0) * 1000)
#         verdict        = ensemble_verdict(model_results)
#         activated      = [FEATURE_LIST[i] for i, v in enumerate(vector) if v == 1.0]
#         return jsonify({
#             "model_results":      model_results,
#             "verdict":            verdict,
#             "infer_ms":           infer_ms,
#             "features_active":    len(activated),
#             "features_total":     len(FEATURE_LIST),
#             "permissions_active": [f for f in activated if not f.startswith("L") and f != "activityCalled"],
#             "apis_active":        [f for f in activated if f.startswith("L")],
#         })
#     except Exception as e:
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# @app.route("/models")
# def models_page():
#     info = {}
#     for name, bundle in MODELS.items():
#         model = bundle["model"]
#         info[name] = {
#             "type":    type(model).__name__,
#             "params":  {k: str(v) for k, v in model.get_params().items()
#                         if k in ["n_estimators","max_depth","learning_rate",
#                                  "n_features_in_","max_features"]},
#             "classes": list(bundle["label_encoder"].classes_),
#         }
#         if hasattr(model, "oob_score_"):
#             info[name]["oob_score"] = round(model.oob_score_, 4)
#     return render_template("models.html", models_info=info, feature_count=len(FEATURE_LIST))


# @app.route("/api/features")
# def api_features():
#     return jsonify({"features": FEATURE_LIST,
#                     "groups": FEATURE_GROUPS,
#                     "count": len(FEATURE_LIST)})


# @app.errorhandler(413)
# def too_large(e):
#     return jsonify({"error": f"File too large. Max {MAX_MB} MB."}), 413


# if __name__ == "__main__":
#     print("\n" + "="*55)
#     print("  APK Malware Detector  -  Flask Interface")
#     print("="*55)
#     print(f"  Models   : {list(MODELS.keys()) or 'None – train first'}")
#     print(f"  Features : {len(FEATURE_LIST)}")
#     print(f"  Routes   : /  /test  /models  /api/features")
#     print("="*55 + "\n")
#     app.run(debug=True, host="0.0.0.0", port=5000)


"""
app.py  -  Flask APK Malware Detection Interface
================================================================
Routes:
  GET  /                    →  APK upload page
  POST /analyze             →  analyze uploaded APK  →  JSON
  GET  /result/<job_id>     →  result page
  GET  /test                →  manual feature tester
  POST /api/test-features   →  run models on manual vector  →  JSON
  GET  /models              →  model info page
  GET  /api/features        →  feature list JSON
================================================================
"""

import os, uuid, json, time, traceback
import numpy as np
from pathlib import Path
from flask   import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import joblib

from apk_extractor import feature_vector, FEATURE_LIST

# ── Config ────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_DIR     = BASE_DIR / "models"
ALLOWED_EXT   = {"apk"}
MAX_MB        = 100

UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"]      = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

JOBS: dict = {}   # in-memory job store

# ── Model loader ──────────────────────────────────────────
MODELS: dict = {}

def load_models():
    specs = {
        "Random Forest": "random_forest/rf_model.pkl",
        "Decision Tree": "decision_tree/dt_model.pkl",
        "XGBoost"      : "xgboost/xgb_model.pkl",
        "AdaBoost"     : "adaboost/adaboost_model.pkl",
        "Stacking"     : "stacking/stacking_model.pkl",
    }
    for name, rel in specs.items():
        full = MODEL_DIR / rel
        if full.exists():
            try:
                MODELS[name] = joblib.load(full)
                print(f"  ✓  Loaded {name}")
            except Exception as e:
                print(f"  ✗  Failed {name}: {e}")
        else:
            print(f"  ⚠  Not found: {full}  (run training script first)")

load_models()

# ── Feature weights from pkl files ───────────────────────
# Extracts feature_importances_ from every loaded model and maps them
# back to the full 241-feature space (zero for features removed by
# VarianceThreshold during training).  Only tree-based models expose
# feature_importances_; XGBoost / Stacking expose it too when their
# internal estimators carry it.  If a model doesn't have it we skip it.
FEATURE_WEIGHTS: dict = {}

def extract_feature_weights():
    for name, bundle in MODELS.items():
        model = bundle["model"]
        vt    = bundle.get("variance_filter")

        if not hasattr(model, "feature_importances_"):
            continue

        imps = model.feature_importances_          # shape = (n_kept_features,)
        mask = vt.get_support() if vt is not None else [True] * len(FEATURE_LIST)

        weights: dict = {}
        ki = 0
        for feat, kept in zip(FEATURE_LIST, mask):
            if kept:
                weights[feat] = round(float(imps[ki]), 6)
                ki += 1
            else:
                weights[feat] = 0.0          # filtered out – zero importance
        FEATURE_WEIGHTS[name] = weights
        print(f"  ✓  Weights extracted  {name}  ({sum(1 for v in weights.values() if v>0)} non-zero)")

extract_feature_weights()

# ── Feature categories (for manual tester) ───────────────
DANGEROUS_PERMS = {
    "READ_SMS","SEND_SMS","RECEIVE_SMS","RECORD_AUDIO",
    "READ_CONTACTS","WRITE_CONTACTS","READ_CALL_LOG","WRITE_CALL_LOG",
    "CAMERA","ACCESS_FINE_LOCATION","ACCESS_COARSE_LOCATION",
    "CALL_PHONE","READ_PHONE_STATE","PROCESS_OUTGOING_CALLS",
    "GET_ACCOUNTS","WRITE_EXTERNAL_STORAGE","READ_EXTERNAL_STORAGE",
    "SYSTEM_ALERT_WINDOW","INSTALL_PACKAGES","DELETE_PACKAGES",
    "RECEIVE_BOOT_COMPLETED","ACCESS_SUPERUSER","MASTER_CLEAR",
    "REBOOT","BRICK","MODIFY_PHONE_STATE",
}

def _categorise_features():
    groups = {
        "Dangerous Permissions": [],
        "Network & Connectivity": [],
        "Storage & Files": [],
        "System & Device": [],
        "Accounts & Sync": [],
        "Media & Sensors": [],
        "Other Permissions": [],
        "Dangerous API Calls": [],
    }
    net  = {"NETWORK","WIFI","INTERNET","BLUETOOTH","NFC","CHANGE_NETWORK","ACCESS_WIFI","ACCESS_NETWORK"}
    stor = {"STORAGE","EXTERNAL_STORAGE","INTERNAL_STORAGE","SDCARD","MOUNT","CLEAR_APP","DELETE_CACHE"}
    sys_ = {"SYSTEM","DEVICE","REBOOT","BRICK","MASTER_CLEAR","INJECT","FACTORY","INSTALL_PACKAGES",
             "DELETE_PACKAGES","CHANGE_COMPONENT","BIND_DEVICE","CHANGE_CONFIG","SET_TIME",
             "RECEIVE_BOOT","SUPERUSER","MODIFY_PHONE","WAKE_LOCK"}
    acc  = {"ACCOUNT","SYNC","AUTH","GET_ACCOUNTS","MANAGE_ACCOUNTS","AUTHENTICATE"}
    med  = {"CAMERA","AUDIO","RECORD","MEDIA","VIBRATE","SENSOR","BODY_SENSOR","FLASHLIGHT","TRANSMIT_IR"}

    for feat in FEATURE_LIST:
        if feat.startswith("L"):
            groups["Dangerous API Calls"].append(feat)
        elif feat in DANGEROUS_PERMS:
            groups["Dangerous Permissions"].append(feat)
        elif any(k in feat for k in net):
            groups["Network & Connectivity"].append(feat)
        elif any(k in feat for k in stor):
            groups["Storage & Files"].append(feat)
        elif any(k in feat for k in sys_):
            groups["System & Device"].append(feat)
        elif any(k in feat for k in acc):
            groups["Accounts & Sync"].append(feat)
        elif any(k in feat for k in med):
            groups["Media & Sensors"].append(feat)
        else:
            groups["Other Permissions"].append(feat)
    return {k: v for k, v in groups.items() if v}

FEATURE_GROUPS = _categorise_features()

# ── Core inference helpers ────────────────────────────────

def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def run_models(vector: list) -> dict:
    X = np.array(vector, dtype=np.float32).reshape(1, -1)
    results = {}
    for name, bundle in MODELS.items():
        try:
            model = bundle["model"]
            le    = bundle["label_encoder"]
            vt    = bundle.get("variance_filter")
            X_in  = vt.transform(X) if vt is not None else X
            proba = model.predict_proba(X_in)[0]
            pred  = le.inverse_transform([int(np.argmax(proba))])[0]
            classes   = list(le.classes_)
            mal_idx   = classes.index("malware") if "malware" in classes else 1
            mal_prob  = float(proba[mal_idx])
            results[name] = {
                "prediction":    pred,
                "malware_prob":  round(mal_prob * 100, 2),
                "goodware_prob": round((1 - mal_prob) * 100, 2),
                "confidence":    round(float(max(proba)) * 100, 2),
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def ensemble_verdict(model_results: dict) -> dict:
    preds, probs = [], []
    for r in model_results.values():
        if "error" not in r:
            preds.append(r["prediction"])
            probs.append(r["malware_prob"])
    if not preds:
        return {"verdict":"unknown","score":0,"risk_level":"Unknown",
                "confidence":0,"mal_votes":0,"good_votes":0,"total_models":0}
    mal_v  = preds.count("malware")
    good_v = preds.count("goodware")
    avg    = round(sum(probs) / len(probs), 2)
    verdict= "malware" if mal_v >= good_v else "goodware"
    conf   = round(max(mal_v, good_v) / len(preds) * 100, 1)
    risk   = ("Critical" if avg >= 80 else "High"   if avg >= 60 else
              "Medium"   if avg >= 40 else "Low"    if avg >= 20 else "Safe")
    return {"verdict": verdict, "score": avg, "risk_level": risk,
            "confidence": conf, "mal_votes": mal_v, "good_votes": good_v,
            "total_models": len(preds)}

# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           model_names=list(MODELS.keys()) or ["No models loaded"],
                           models_loaded=len(MODELS))


@app.route("/analyze", methods=["POST"])
def analyze():
    if "apk_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["apk_file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Only .apk files accepted"}), 400
    if not MODELS:
        return jsonify({"error": "No models loaded – run training scripts first"}), 503

    job_id   = str(uuid.uuid4())
    filename = secure_filename(f.filename)
    apk_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{filename}")
    f.save(apk_path)

    try:
        t0                   = time.time()
        vec, meta            = feature_vector(apk_path)
        extract_ms           = round((time.time() - t0) * 1000)
        t1                   = time.time()
        model_results        = run_models(vec)
        infer_ms             = round((time.time() - t1) * 1000)
        verdict              = ensemble_verdict(model_results)
        active               = [FEATURE_LIST[i] for i, v in enumerate(vec) if v == 1.0]
        permissions_active   = [f for f in active if not f.startswith("L") and f != "activityCalled"]
        apis_active          = [f for f in active if f.startswith("L")]

        result = {
            "job_id": job_id, "filename": filename,
            "apk_size_kb": meta["_apk_size_kb"], "dex_count": meta["_dex_count"],
            "extract_error": meta["_error"], "extract_ms": extract_ms, "infer_ms": infer_ms,
            "features_total": len(FEATURE_LIST), "features_active": len(active),
            "permissions_active": permissions_active, "apis_active": apis_active,
            "model_results": model_results, "verdict": verdict,
        }
        JOBS[job_id] = result
        try: os.remove(apk_path)
        except: pass
        return jsonify(result)

    except Exception as e:
        try: os.remove(apk_path)
        except: pass
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/result/<job_id>")
def result_page(job_id):
    job = JOBS.get(job_id)
    if not job:
        return redirect(url_for("index"))
    return render_template("result.html", result=job)


@app.route("/test")
def test_page():
    return render_template("test.html",
                           feature_groups=FEATURE_GROUPS,
                           feature_list=FEATURE_LIST,
                           dangerous_perms=list(DANGEROUS_PERMS),
                           models_loaded=len(MODELS),
                           model_names=list(MODELS.keys()),
                           total_features=len(FEATURE_LIST),
                           feature_weights=FEATURE_WEIGHTS)        # ← NEW


@app.route("/api/test-features", methods=["POST"])
def api_test_features():
    if not MODELS:
        return jsonify({"error": "No models loaded"}), 503
    try:
        data   = request.get_json(force=True) or {}
        active = {k: int(bool(v)) for k, v in data.items() if k in set(FEATURE_LIST)}
        vector = [float(active.get(f, 0)) for f in FEATURE_LIST]
        t0             = time.time()
        model_results  = run_models(vector)
        infer_ms       = round((time.time() - t0) * 1000)
        verdict        = ensemble_verdict(model_results)
        activated      = [FEATURE_LIST[i] for i, v in enumerate(vector) if v == 1.0]
        return jsonify({
            "model_results":      model_results,
            "verdict":            verdict,
            "infer_ms":           infer_ms,
            "features_active":    len(activated),
            "features_total":     len(FEATURE_LIST),
            "permissions_active": [f for f in activated if not f.startswith("L") and f != "activityCalled"],
            "apis_active":        [f for f in activated if f.startswith("L")],
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/models")
def models_page():
    info = {}
    for name, bundle in MODELS.items():
        model = bundle["model"]
        info[name] = {
            "type":    type(model).__name__,
            "params":  {k: str(v) for k, v in model.get_params().items()
                        if k in ["n_estimators","max_depth","learning_rate",
                                 "n_features_in_","max_features"]},
            "classes": list(bundle["label_encoder"].classes_),
        }
        if hasattr(model, "oob_score_"):
            info[name]["oob_score"] = round(model.oob_score_, 4)
    return render_template("models.html", models_info=info, feature_count=len(FEATURE_LIST))


@app.route("/api/features")
def api_features():
    return jsonify({"features": FEATURE_LIST,
                    "groups": FEATURE_GROUPS,
                    "count": len(FEATURE_LIST)})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {MAX_MB} MB."}), 413


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  APK Malware Detector  -  Flask Interface")
    print("="*55)
    print(f"  Models   : {list(MODELS.keys()) or 'None – train first'}")
    print(f"  Features : {len(FEATURE_LIST)}")
    print(f"  Routes   : /  /test  /models  /api/features")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)