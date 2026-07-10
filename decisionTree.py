"""
================================================================
  Android Malware Detection  -  Decision Tree Classifier
================================================================
Dataset  : TUANDROMD.csv  (4464 samples, 241 binary features)
Imbalance strategy : class_weight="balanced"  (NO SMOTE)
Fine-tuning        : GridSearchCV  (5-fold CV)
Output : outputs/decision_tree/dt_model.pkl
================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.tree              import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection   import (train_test_split, StratifiedKFold,
                                        GridSearchCV, cross_validate,
                                        learning_curve)
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (accuracy_score, precision_score,
                                        recall_score, f1_score, roc_auc_score,
                                        matthews_corrcoef, confusion_matrix,
                                        classification_report, roc_curve)
import joblib

DATA_PATH = "./data/TUANDROMD.csv"
OUT       = "outputs/decision_tree"
os.makedirs(OUT, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────
print("="*60)
print("  DECISION TREE  |  Android Malware Detection")
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

# ── 2. GridSearchCV ───────────────────────────────────────
print("[1/5]  GridSearchCV  (5-fold) ...")
param_grid = {
    "criterion"        : ["gini", "entropy"],
    "max_depth"        : [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "ccp_alpha"        : [0.0, 0.001, 0.005],
    "class_weight"     : ["balanced"],
}
cv5    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, scoring="f1_macro",
    cv=cv5, n_jobs=-1, verbose=1)
search.fit(X_tr, y_tr)
print(f"\n  Best params : {search.best_params_}")
print(f"  Best CV F1  : {search.best_score_:.4f}\n")

# ── 3. Final model ────────────────────────────────────────
print("[2/5]  Final model ...")
clf = search.best_estimator_
print(f"  Depth  : {clf.get_depth()}")
print(f"  Leaves : {clf.get_n_leaves()}")

# ── 4. Cross-validation + learning curve ─────────────────
print("\n[3/5]  5-fold CV + learning curve ...")
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
    train_sizes=np.linspace(0.10, 1.0, 8), n_jobs=-1)
lc_tr_m = lc_tr.mean(1); lc_tr_s = lc_tr.std(1)
lc_cv_m = lc_cv.mean(1); lc_cv_s = lc_cv.std(1)

# ── 5. Test evaluation ────────────────────────────────────
print("\n[4/5]  Test evaluation ...")
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

# ── 6. Plots ──────────────────────────────────────────────
print("[5/5]  Saving plots ...")
cm = confusion_matrix(y_te, y_pred)
fpr_, tpr_, _ = roc_curve(y_te, y_proba)
fi  = clf.feature_importances_
top20 = np.argsort(fi)[-20:]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0,0], cbar=False, annot_kws={"size":13})
axes[0,0].set_title("Confusion Matrix (Test)")
axes[0,0].set_xlabel("Predicted"); axes[0,0].set_ylabel("Actual")

# ROC
axes[0,1].plot(fpr_, tpr_, color="seagreen", lw=2, label=f"AUC={auc:.4f}")
axes[0,1].plot([0,1],[0,1], "k--", lw=1)
axes[0,1].set_xlabel("FPR"); axes[0,1].set_ylabel("TPR")
axes[0,1].set_title("ROC Curve"); axes[0,1].legend(); axes[0,1].grid(alpha=.3)

# Learning curve
axes[1,0].plot(tsz, lc_tr_m, "o-", color="seagreen", label="Train F1")
axes[1,0].fill_between(tsz, lc_tr_m-lc_tr_s, lc_tr_m+lc_tr_s, alpha=.12, color="seagreen")
axes[1,0].plot(tsz, lc_cv_m, "s-", color="darkorange", label="CV F1")
axes[1,0].fill_between(tsz, lc_cv_m-lc_cv_s, lc_cv_m+lc_cv_s, alpha=.12, color="darkorange")
axes[1,0].set_xlabel("Train size"); axes[1,0].set_ylabel("F1 Macro")
axes[1,0].set_title("Learning Curve"); axes[1,0].legend()
axes[1,0].set_ylim(0, 1.05); axes[1,0].grid(alpha=.3)

# Feature importance
axes[1,1].barh(range(20), fi[top20], color="seagreen", alpha=.85)
axes[1,1].set_yticks(range(20))
axes[1,1].set_yticklabels([str(i) for i in top20], fontsize=7)
axes[1,1].set_xlabel("Gini Importance")
axes[1,1].set_title("Top-20 Feature Importances")

plt.suptitle(f"Decision Tree  |  F1={f1:.4f}  AUC={auc:.4f}  depth={clf.get_depth()}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/dashboard.png", dpi=150)
plt.close()

# Shallow tree visualisation
fig2, ax2 = plt.subplots(figsize=(20, 8))
plot_tree(clf, max_depth=3, filled=True, class_names=le.classes_,
          feature_names=[str(i) for i in range(X.shape[1])],
          ax=ax2, fontsize=7)
ax2.set_title(f"Decision Tree (showing depth ≤ 3 of {clf.get_depth()})")
plt.tight_layout()
plt.savefig(f"{OUT}/tree_visual.png", dpi=120)
plt.close()

joblib.dump({"model": clf, "label_encoder": le, "variance_filter": vt},
            f"{OUT}/dt_model.pkl")
print(f"\n  Plots  →  {OUT}/dashboard.png")
print(f"  Model  →  {OUT}/dt_model.pkl")
print(f"\n{'='*60}")
print(f"  DONE  |  F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}")
print(f"{'='*60}")