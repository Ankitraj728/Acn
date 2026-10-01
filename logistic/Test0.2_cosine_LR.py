#!/usr/bin/env python3
"""Logistic Regression | Test size 0.2 | Kernel PCA kernel=cosine, n_components=10
CIC-IDS2017 (Tue+Wed+Thu-morning) 1000-row stratified sample.
Run in any IDE (Spyder / VS Code / PyCharm / Jupyter): open the file and press Run, no arguments needed.
Saves next to this file: 'Test0.2_cosine_LR.jpg' - one image with the console output (confusion matrix, metrics, classification report) plus charts."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TEST_SIZE, KERNEL, N_COMP, ALG, ALG_NAME = 0.2, "cosine", 10, "LR", "Logistic Regression"
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # Jupyter / interactive console
    HERE = os.getcwd()
def _find_root():                      # folder that contains data/sample.npz (works from any working directory)
    for p in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", ".."), os.getcwd()):
        if os.path.exists(os.path.join(p, "data", "sample.npz")):
            return os.path.abspath(p)
    raise FileNotFoundError("data/sample.npz not found - run prepare_data.py or keep the repo folder structure")
ROOT = _find_root()
tag = f"Test{TEST_SIZE}_{KERNEL}_{ALG}"

# 1. LOAD cached, cleaned sample (see prepare_data.py)
d = np.load(os.path.join(ROOT, "data", "sample.npz"), allow_pickle=True)
X, y, class_names = d["X"], d["y"], [str(c).replace("�", "-") for c in d["classes"]]

# 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

# 3. SCALE (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

# 4. DIMENSIONALITY REDUCTION: Kernel PCA (fit on train only; gamma = sklearn default 1/n_features)
kpca = KernelPCA(n_components=N_COMP, kernel=KERNEL, random_state=0)
X_train = kpca.fit_transform(X_train); X_test = kpca.transform(X_test)

# 5. CLASSIFIER
classifier = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# 6. METRICS
labels = np.arange(len(class_names))
cm = confusion_matrix(y_test, y_pred, labels=labels)
res = dict(test_size=TEST_SIZE, kernel=KERNEL, n_components=N_COMP, algorithm=ALG, model=ALG_NAME,
           n_train=int(len(y_train)), n_test=int(len(y_test)),
           accuracy=accuracy_score(y_test, y_pred),
           precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
           recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
           f1=f1_score(y_test, y_pred, average="weighted", zero_division=0))
report = classification_report(y_test, y_pred, labels=labels, target_names=class_names, zero_division=0)
lines = ["=" * 60, f"{ALG_NAME} | test size={TEST_SIZE} | Kernel PCA {KERNEL}, nComp={N_COMP}", "=" * 60,
         f"Training samples: {len(y_train)}   Testing samples: {len(y_test)}", "", "CONFUSION MATRIX:", np.array2string(cm), ""]
for k in ("accuracy", "precision", "recall", "f1"): lines.append(f"{k.capitalize():10s}: {res[k]:.4f}")
lines += ["", "CLASSIFICATION REPORT:", report]
console_output = "\n".join(lines); print(console_output)

# 7. VISUAL OUTPUT: ONE report-style image (banner, KPI cards, charts, console output) - real results only
from matplotlib.patches import FancyBboxPatch, Rectangle
short = [c.replace("Web Attack - ", "Web ").replace("DoS ", "DoS-") for c in class_names]
rep = classification_report(y_test, y_pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True)
cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)          # row-normalised = per-class recall
INK, MUTED, BG, ACC = "#1f2937", "#6b7280", "#faf7f2", "#0f766e"
fig = plt.figure(figsize=(20, 13.2), dpi=105, facecolor=BG)
fig.add_artist(Rectangle((0, 0.925), 1, 0.075, transform=fig.transFigure, color=ACC, zorder=0))
fig.text(0.03, 0.968, f"{ALG_NAME}", fontsize=24, fontweight="bold", color="white", va="center")
fig.text(0.03, 0.937, f"Network intrusion detection  |  CIC-IDS2017  |  Kernel PCA: {KERNEL} ({N_COMP} components)  |  test size {TEST_SIZE}",
         fontsize=12.5, color="#ccfbf1", va="center")
fig.text(0.97, 0.952, f"run: {tag}", fontsize=12, color="white", ha="right", va="center", family="monospace")

# KPI cards
cards = [("ACCURACY", res["accuracy"], "#2563eb"), ("PRECISION (weighted)", res["precision"], "#059669"),
         ("RECALL (weighted)", res["recall"], "#d97706"), ("F1 (weighted)", res["f1"], "#7c3aed")]
for i, (lab, v, col) in enumerate(cards):
    x0 = 0.03 + i * 0.1925
    fig.add_artist(FancyBboxPatch((x0, 0.815), 0.18, 0.09, boxstyle="round,pad=0.002,rounding_size=0.008", transform=fig.transFigure,
                                  facecolor="white", edgecolor=col, linewidth=2))
    fig.text(x0 + 0.012, 0.887, lab, fontsize=10.5, color=MUTED, fontweight="bold")
    fig.text(x0 + 0.012, 0.842, f"{v * 100:.2f}%", fontsize=26, color=col, fontweight="bold")
fig.add_artist(FancyBboxPatch((0.80, 0.815), 0.17, 0.09, boxstyle="round,pad=0.002,rounding_size=0.008", transform=fig.transFigure,
                              facecolor="white", edgecolor="#9ca3af", linewidth=1.5))
fig.text(0.812, 0.887, "SAMPLES", fontsize=10.5, color=MUTED, fontweight="bold")
fig.text(0.812, 0.848, f"train {res['n_train']}  |  test {res['n_test']}", fontsize=15, color=INK, fontweight="bold")
fig.text(0.812, 0.826, f"{len(class_names)} classes", fontsize=10.5, color=MUTED)

gs = fig.add_gridspec(1, 2, left=0.05, right=0.97, top=0.78, bottom=0.47, wspace=0.30, width_ratios=[1.3, 1])
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
for ax in (ax1, ax2): ax.set_facecolor("white")
im = ax1.imshow(cmn, cmap="YlOrBr", vmin=0, vmax=1, aspect="auto")
ax1.set_xticks(labels); ax1.set_yticks(labels); ax1.set_xticklabels(short, rotation=40, ha="right", fontsize=8); ax1.set_yticklabels(short, fontsize=8)
ax1.set_title("Confusion matrix (colour = share of each true class, number = count)", loc="left", fontsize=11.5, color=INK, fontweight="bold")
ax1.set_xlabel("Predicted class", color=MUTED); ax1.set_ylabel("True class", color=MUTED)
for a_ in labels:
    for b_ in labels:
        if cm[a_, b_]: ax1.text(b_, a_, cm[a_, b_], ha="center", va="center", fontsize=8, color="white" if cmn[a_, b_] > 0.6 else INK)
f1s = [rep[c]["f1-score"] for c in class_names]; sup = [int(rep[c]["support"]) for c in class_names]
order = np.argsort(f1s)
cols = ["#dc2626" if f1s[i] < 0.34 else "#f59e0b" if f1s[i] < 0.67 else "#16a34a" for i in order]
ax2.barh(range(len(order)), [f1s[i] for i in order], color=cols)
ax2.set_yticks(range(len(order))); ax2.set_yticklabels([short[i] for i in order], fontsize=8.5); ax2.set_xlim(0, 1.25)
ax2.set_title("F1 score per class (n = test rows)", loc="left", fontsize=11.5, color=INK, fontweight="bold")
for k, i in enumerate(order): ax2.text(f1s[i] + 0.02, k, f"{f1s[i]:.2f}   n={sup[i]}", va="center", fontsize=8.5, color=INK, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1))
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.axvline(res["f1"], color=ACC, ls="--", lw=1.2, label=f"weighted F1 = {res['f1']:.3f}"); ax2.legend(loc="lower right", fontsize=9, frameon=False)

# console output panel (same text that is printed to the console)
fig.add_artist(FancyBboxPatch((0.03, 0.025), 0.94, 0.30, boxstyle="round,pad=0.002,rounding_size=0.006", transform=fig.transFigure,
                              facecolor="#fffdf8", edgecolor="#d6d3d1", linewidth=1.2))
fig.text(0.04, 0.305, "CONSOLE OUTPUT", fontsize=10.5, color=ACC, fontweight="bold")
split_at = lines.index("CLASSIFICATION REPORT:")
fig.text(0.04, 0.288, "\n".join(lines[:split_at]), family="monospace", fontsize=8.8, color=INK, va="top")
fig.text(0.50, 0.288, "\n".join(lines[split_at:]), family="monospace", fontsize=8.8, color=INK, va="top")
plt.savefig(os.path.join(HERE, tag + ".jpg"), format="jpg", bbox_inches="tight", facecolor=BG)
if os.environ.get("ACN_NO_SHOW") != "1": plt.show()   # IDEs display the image; run_all.py disables this
plt.close()

os.makedirs(os.path.join(ROOT, "results", "json"), exist_ok=True)
with open(os.path.join(ROOT, "results", "json", tag + ".json"), "w") as f: json.dump(res, f)
