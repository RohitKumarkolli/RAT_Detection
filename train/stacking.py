"""
================================================================
  Android Malware Detection  -  Stacking Ensemble
================================================================
Dataset  : TUANDROMD.csv  (4464 samples, 241 binary features)
Imbalance strategy : class_weight / scale_pos_weight on all base
                     learners  (NO SMOTE)

Architecture:
  Level-0 base learners (5):
    1. Random Forest      (class_weight=balanced)
    2. Extra Trees        (class_weight=balanced)
    3. XGBoost            (scale_pos_weight)
    4. Logistic Reg.      (class_weight=balanced, StandardScaler)
    5. KNN                (StandardScaler)
  Level-1 meta-learner:
    Logistic Regression   (C tuned via GridSearchCV)

Output : outputs/stacking/stacking_model.pkl
================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.ensemble          import (StackingClassifier,
                                        RandomForestClassifier,
                                        ExtraTreesClassifier)
from sklearn.linear_model      import LogisticRegression
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection   import (train_test_split, StratifiedKFold,
                                        GridSearchCV, cross_validate,
                                        cross_val_score, learning_curve)
from sklearn.metrics           import (accuracy_score, precision_score,
                                        recall_score, f1_score, roc_auc_score,
                                        matthews_corrcoef, confusion_matrix,
                                        classification_report, roc_curve)
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("  ⚠  XGBoost not found – using GradientBoosting instead")

DATA_PATH = "./data/TUANDROMD.csv"
OUT       = "outputs/stacking"
os.makedirs(OUT, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────
print("="*60)
print("  STACKING ENSEMBLE  |  Android Malware Detection")
print("="*60)
df = pd.read_csv(DATA_PATH)
df = df[df["Label"].notna()].reset_index(drop=True)
print(f"\n  Samples : {len(df)}")
print(f"  Classes :\n{df['Label'].value_counts().to_string()}")

le = LabelEncoder()
y  = le.fit_transform(df["Label"])
X  = df.drop(columns=["Label"]).fillna(0).astype(np.float32)

vt = VarianceThreshold(threshold=0.0)
X  = vt.fit_transform(X)
print(f"\n  Features : 241  →  {X.shape[1]}  (after zero-variance removal)")

counts          = np.bincount(y)
spw             = counts[0] / counts[1]     # scale_pos_weight for XGBoost
print(f"  scale_pos_weight : {spw:.4f}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  Train : {len(y_tr)}   Test : {len(y_te)}\n")

# ── 2. Define base learners ───────────────────────────────
print("[1/6]  Defining base learners ...")
if HAS_XGB:
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw,
                         use_label_encoder=False, eval_metric="logloss",
                         random_state=42, n_jobs=-1, tree_method="hist")
else:
    xgb = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                      learning_rate=0.1, random_state=42)

base_learners = [
    ("random_forest", RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", random_state=42, n_jobs=-1)),
    ("extra_trees", ExtraTreesClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", random_state=42, n_jobs=-1)),
    ("xgboost", xgb),
    ("logistic", Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, random_state=42))])),
    ("knn", Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])),
]
for name, _ in base_learners:
    print(f"    • {name}")

# ── 3. Evaluate each base learner ─────────────────────────
print("\n[2/6]  5-fold CV per base learner ...")
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_cv = {}
print(f"\n  {'Learner':<18} {'F1 Mean':>10}  {'F1 Std':>9}  {'AUC':>9}")
print("  " + "-"*52)
for name, est in base_learners:
    f1_s  = cross_val_score(est, X_tr, y_tr, cv=cv5, scoring="f1_macro",  n_jobs=-1)
    auc_s = cross_val_score(est, X_tr, y_tr, cv=cv5, scoring="roc_auc",   n_jobs=-1)
    base_cv[name] = {"f1": f1_s.mean(), "f1_std": f1_s.std(), "auc": auc_s.mean()}
    print(f"  {name:<18} {f1_s.mean():>10.4f}  {f1_s.std():>9.4f}  {auc_s.mean():>9.4f}")

# ── 4. Stacking + meta-learner tuning ─────────────────────
print("\n[3/6]  Stacking + meta-learner C tuning ...")
stack = StackingClassifier(
    estimators      = base_learners,
    final_estimator = LogisticRegression(max_iter=1000, random_state=42,
                                          class_weight="balanced"),
    cv              = StratifiedKFold(5, shuffle=True, random_state=42),
    passthrough     = False,
    n_jobs          = -1)

meta_search = GridSearchCV(
    stack, {"final_estimator__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
    scoring="f1_macro", cv=cv5, n_jobs=-1, verbose=1,
    return_train_score=True)
meta_search.fit(X_tr, y_tr)
clf = meta_search.best_estimator_
best_C = meta_search.best_params_["final_estimator__C"]
print(f"\n  Best meta C : {best_C}")
print(f"  Best CV F1  : {meta_search.best_score_:.4f}\n")

# ── 5. Cross-validation + learning curve ─────────────────
print("[4/6]  Full stack 5-fold CV + learning curve ...")
cv_res = cross_validate(
    clf, X_tr, y_tr, cv=cv5,
    scoring={"accuracy":"accuracy","f1_macro":"f1_macro",
             "roc_auc":"roc_auc","precision":"precision_macro",
             "recall":"recall_macro"},
    return_train_score=True, n_jobs=-1)

print(f"\n  {'Metric':<18} {'Train':>9}  {'CV':>9}  {'Gap':>8}  Status")
print("  " + "-"*56)
for m in ["accuracy","f1_macro","roc_auc","precision","recall"]:
    tr = cv_res[f"train_{m}"].mean()
    cv = cv_res[f"test_{m}"].mean()
    g  = tr - cv
    flag = "  ⚠  overfit" if g > 0.05 else "  ✓  good"
    print(f"  {m:<18} {tr:>9.4f}  {cv:>9.4f}  {g:>8.4f}{flag}")

# Learning curve (3-fold to save time)
tsz, lc_tr, lc_cv = learning_curve(
    clf, X_tr, y_tr,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring="f1_macro", train_sizes=np.linspace(0.20, 1.0, 5), n_jobs=-1)
lc_tr_m = lc_tr.mean(1); lc_tr_s = lc_tr.std(1)
lc_cv_m = lc_cv.mean(1); lc_cv_s = lc_cv.std(1)

# ── 6. Test evaluation ────────────────────────────────────
print("\n[5/6]  Test evaluation ...")
y_pred  = clf.predict(X_te)
y_proba = clf.predict_proba(X_te)[:, 1]
acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, average="macro")
rec  = recall_score(y_te, y_pred, average="macro")
f1   = f1_score(y_te, y_pred, average="macro")
auc  = roc_auc_score(y_te, y_proba)
mcc  = matthews_corrcoef(y_te, y_pred)
print(f"\n  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Macro  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  MCC       : {mcc:.4f}")
print("\n" + classification_report(y_te, y_pred, target_names=le.classes_))

# Per base learner test scores
print("  Per-base-learner on test:")
base_test = {}
for name, _ in base_learners:
    est = clf.named_estimators_[name]
    bp  = est.predict(X_te)
    bpr = est.predict_proba(X_te)[:, 1]
    base_test[name] = {
        "f1":  round(f1_score(y_te, bp, average="macro"), 4),
        "auc": round(roc_auc_score(y_te, bpr), 4)}
    print(f"    {name:<18}  F1={base_test[name]['f1']:.4f}  AUC={base_test[name]['auc']:.4f}")
print(f"    {'STACKING':<18}  F1={f1:.4f}  AUC={auc:.4f}  ← ensemble")

# ── 7. Plots ──────────────────────────────────────────────
print("\n[6/6]  Saving plots ...")
cm = confusion_matrix(y_te, y_pred)
fpr_, tpr_, _ = roc_curve(y_te, y_proba)
names_all  = list(base_cv.keys()) + ["STACKING"]
f1_all     = [base_cv[n]["f1"]  for n in base_cv] + [cv_res["test_f1_macro"].mean()]
f1_std_all = [base_cv[n]["f1_std"] for n in base_cv] + [cv_res["test_f1_macro"].std()]
auc_all    = [base_cv[n]["auc"] for n in base_cv] + [cv_res["test_roc_auc"].mean()]
colors     = ["steelblue"]*5 + ["gold"]

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.36)

# Base learner comparison
ax = fig.add_subplot(gs[0, 0])
bars = ax.bar(names_all, f1_all, yerr=f1_std_all, capsize=4,
              color=colors, alpha=.88, edgecolor="white")
ax.set_ylim(0, 1.12); ax.set_ylabel("CV F1 Macro")
ax.set_title("Base Learners vs Stacking")
ax.set_xticklabels(names_all, rotation=20, ha="right", fontsize=8)
ax.grid(axis="y", alpha=.3)
for bar, val in zip(bars, f1_all):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
            f"{val:.3f}", ha="center", fontsize=8)

# Learning curve
ax = fig.add_subplot(gs[0, 1])
ax.plot(tsz, lc_tr_m, "o-", color="steelblue", label="Train F1")
ax.fill_between(tsz, lc_tr_m-lc_tr_s, lc_tr_m+lc_tr_s, alpha=.12, color="steelblue")
ax.plot(tsz, lc_cv_m, "s-", color="darkorange", label="CV F1")
ax.fill_between(tsz, lc_cv_m-lc_cv_s, lc_cv_m+lc_cv_s, alpha=.12, color="darkorange")
ax.set_xlabel("Train size"); ax.set_ylabel("F1 Macro")
ax.set_title("Learning Curve"); ax.legend(); ax.set_ylim(0, 1.05); ax.grid(alpha=.3)

# CV per-fold
ax = fig.add_subplot(gs[0, 2])
xi = np.arange(5)
ax.bar(xi-.2, cv_res["train_f1_macro"], .4, label="Train", color="steelblue", alpha=.85)
ax.bar(xi+.2, cv_res["test_f1_macro"],  .4, label="CV",    color="darkorange", alpha=.85)
ax.set_xticks(xi); ax.set_xticklabels([f"F{i+1}" for i in range(5)])
ax.set_ylabel("F1 Macro"); ax.set_title("5-Fold CV F1"); ax.set_ylim(0, 1.12)
ax.legend(); ax.grid(axis="y", alpha=.3)

# Confusion matrix
ax = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, cbar=False)
ax.set_title("Confusion Matrix (Test)"); ax.set_xlabel("Pred"); ax.set_ylabel("Actual")

# ROC comparison
ax = fig.add_subplot(gs[1, 1])
ax.plot(fpr_, tpr_, color="gold", lw=2.5, label=f"Stacking AUC={auc:.4f}")
cols_b = ["steelblue","seagreen","darkorange","mediumpurple","coral"]
for (nm, scores_), col in zip(base_test.items(), cols_b):
    try:
        est = clf.named_estimators_[nm]
        fp, tp, _ = roc_curve(y_te, est.predict_proba(X_te)[:, 1])
        ax.plot(fp, tp, lw=1, ls="--", color=col, alpha=.7,
                label=f"{nm} {scores_['auc']:.3f}")
    except Exception:
        pass
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC: Stack vs Base Learners"); ax.legend(fontsize=7); ax.grid(alpha=.3)

# AUC comparison
ax = fig.add_subplot(gs[1, 2])
bars = ax.bar(names_all, auc_all, color=colors, alpha=.88, edgecolor="white")
ax.set_ylim(0, 1.08); ax.set_ylabel("CV ROC-AUC")
ax.set_title("AUC Comparison")
ax.set_xticklabels(names_all, rotation=20, ha="right", fontsize=8)
ax.grid(axis="y", alpha=.3)
for bar, val in zip(bars, auc_all):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003,
            f"{val:.3f}", ha="center", fontsize=8)

fig.suptitle(f"Stacking Ensemble  |  F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}",
             fontsize=12, fontweight="bold")
plt.savefig(f"{OUT}/dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

joblib.dump({"model": clf, "label_encoder": le, "variance_filter": vt},
            f"{OUT}/stacking_model.pkl")
print(f"\n  Plots  →  {OUT}/dashboard.png")
print(f"  Model  →  {OUT}/stacking_model.pkl")
print(f"\n{'='*60}")
print(f"  DONE  |  F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}")
print(f"{'='*60}")