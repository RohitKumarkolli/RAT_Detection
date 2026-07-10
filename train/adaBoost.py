"""
================================================================
  Android Malware Detection  -  AdaBoost Classifier
================================================================
Dataset  : TUANDROMD.csv  (4464 samples, 241 binary features)
Imbalance strategy : class_weight="balanced" on base estimator  (NO SMOTE)
Fine-tuning        : RandomizedSearchCV  (20 iter, 5-fold CV)
Key plot           : Staged error curve (generalisation per round)
Output : outputs/adaboost/adaboost_model.pkl
================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.ensemble          import AdaBoostClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection   import (train_test_split, StratifiedKFold,
                                        RandomizedSearchCV, cross_validate,
                                        learning_curve)
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (accuracy_score, precision_score,
                                        recall_score, f1_score, roc_auc_score,
                                        matthews_corrcoef, confusion_matrix,
                                        classification_report, roc_curve)
import joblib

DATA_PATH = "./data/TUANDROMD.csv"
OUT       = "models/adaboost"
os.makedirs(OUT, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────
print("="*60)
print("  ADABOOST  |  Android Malware Detection")
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

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  Train : {len(y_tr)}   Test : {len(y_te)}\n")

# ── 2. Hyperparameter search ──────────────────────────────
print("[1/6]  RandomizedSearchCV  (20 iter, 5-fold) ...")
param_dist = {
    "n_estimators"             : [50, 100, 200, 300, 500],
    "learning_rate"            : [0.01, 0.05, 0.1, 0.5, 1.0],
    "estimator__max_depth"     : [1, 2, 3, 4],
    "estimator__class_weight"  : ["balanced"],
    "estimator__min_samples_leaf": [1, 2, 5],
}
cv5    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base   = DecisionTreeClassifier(random_state=42)
search = RandomizedSearchCV(
    AdaBoostClassifier(estimator=base, algorithm="SAMME", random_state=42),
    param_dist, n_iter=20, scoring="f1_macro",
    cv=cv5, n_jobs=-1, random_state=42, verbose=1)
search.fit(X_tr, y_tr)
bp = search.best_params_
print(f"\n  Best params : {bp}")
print(f"  Best CV F1  : {search.best_score_:.4f}\n")

# ── 3. Final model ────────────────────────────────────────
print("[2/6]  Training final model ...")
best_base = DecisionTreeClassifier(
    max_depth         = bp["estimator__max_depth"],
    class_weight      = bp["estimator__class_weight"],
    min_samples_leaf  = bp["estimator__min_samples_leaf"],
    random_state=42)
clf = AdaBoostClassifier(
    estimator     = best_base,
    n_estimators  = bp["n_estimators"],
    learning_rate = bp["learning_rate"],
    algorithm     = "SAMME",
    random_state  = 42)
clf.fit(X_tr, y_tr)
print(f"  Estimators used : {clf.estimators_}")

# ── 4. Staged error curve ─────────────────────────────────
print("\n[3/6]  Staged error curve ...")
tr_err, te_err = [], []
for tr_pred, te_pred in zip(clf.staged_predict(X_tr), clf.staged_predict(X_te)):
    tr_err.append(1.0 - accuracy_score(y_tr, tr_pred))
    te_err.append(1.0 - accuracy_score(y_te, te_pred))
best_rnd = int(np.argmin(te_err)) + 1
print(f"  Min test error : {min(te_err):.4f}  at round {best_rnd}")
print(f"  Final gap      : {te_err[-1] - tr_err[-1]:.4f}")

# ── 5. Cross-validation + learning curve ─────────────────
print("\n[4/6]  5-fold CV + learning curve ...")
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

tsz, lc_tr, lc_cv = learning_curve(
    clf, X_tr, y_tr, cv=cv5, scoring="f1_macro",
    train_sizes=np.linspace(0.10, 1.0, 7), n_jobs=-1)
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

# ── 7. Plots ──────────────────────────────────────────────
print("[6/6]  Saving plots ...")
cm = confusion_matrix(y_te, y_pred)
fpr_, tpr_, _ = roc_curve(y_te, y_proba)
rnd = list(range(1, len(tr_err)+1))

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.36)

# Staged error curve
ax = fig.add_subplot(gs[0, 0])
ax.plot(rnd, tr_err, color="steelblue",  lw=1.5, label="Train error")
ax.plot(rnd, te_err, color="darkorange", lw=1.5, label="Test error")
ax.axvline(best_rnd, color="green", ls="--", lw=1.2,
           label=f"Best round={best_rnd} ({min(te_err):.4f})")
ax.set_xlabel("Boosting Round"); ax.set_ylabel("Error (1-Acc)")
ax.set_title("Staged Error Curve"); ax.legend(); ax.grid(alpha=.3)

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
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, cbar=False)
ax.set_title("Confusion Matrix (Test)"); ax.set_xlabel("Pred"); ax.set_ylabel("Actual")

# ROC
ax = fig.add_subplot(gs[1, 1])
ax.plot(fpr_, tpr_, color="darkorange", lw=2, label=f"AUC={auc:.4f}")
ax.plot([0,1],[0,1], "k--", lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
ax.legend(); ax.grid(alpha=.3)

# Train/CV metric bars
ax = fig.add_subplot(gs[1, 2])
mlabels = ["Accuracy","F1","AUC","Precision","Recall"]
mkeys   = ["accuracy","f1_macro","roc_auc","precision","recall"]
xm = np.arange(5)
ax.bar(xm-.2, [cv_res[f"train_{m}"].mean() for m in mkeys], .4,
       label="Train", color="steelblue", alpha=.85)
ax.bar(xm+.2, [cv_res[f"test_{m}"].mean()  for m in mkeys], .4,
       label="CV",    color="darkorange", alpha=.85)
ax.set_xticks(xm); ax.set_xticklabels(mlabels, rotation=15, ha="right", fontsize=8)
ax.set_ylim(0, 1.15); ax.set_title("Train vs CV Metrics")
ax.legend(); ax.grid(axis="y", alpha=.3)

fig.suptitle(f"AdaBoost  |  F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}",
             fontsize=12, fontweight="bold")
plt.savefig(f"{OUT}/dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

joblib.dump({"model": clf, "label_encoder": le, "variance_filter": vt},
            f"{OUT}/adaboost_model.pkl")
print(f"\n  Plots  →  {OUT}/dashboard.png")
print(f"  Model  →  {OUT}/adaboost_model.pkl")
print(f"\n{'='*60}")
print(f"  DONE  |  F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}")
print(f"{'='*60}")