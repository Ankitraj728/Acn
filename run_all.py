"""Runs all 75 scripts, then builds results/model_performance_comparison.xlsx:
 - 'Algorithm by Kernel'  : Algorithm | Kernel PCA | nComp | (Accuracy, Precision, Recall, F1) for each test size
 - 'Comparison by Split'  : one row per run (Model, kernel, nComp, split, train/test counts, metrics)
 - '<ALG> accuracy / f1'  : per-algorithm grids (rows kernel+nComp, columns test size)
 - 'Best per Algorithm'   : best run for each algorithm"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results/logs", exist_ok=True)
def run(p):
    with open("results/logs/" + os.path.basename(p)[:-3] + ".txt", "w") as f:
        subprocess.run([sys.executable, p], stdout=f, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "ACN_NO_SHOW": "1", "MPLBACKEND": "Agg"})
scripts = sorted(glob.glob("*/Test*.py"))
with ThreadPoolExecutor(4) as ex: list(ex.map(run, scripts))
df = pd.DataFrame([json.load(open(f)) for f in glob.glob("results/json/*.json")])
A = ["LR", "KNN", "SVM", "DT", "RF"]; K = ["sigmoid", "cosine", "linear", "rbf", "poly"]
df["a"] = df.algorithm.map(A.index); df["k"] = df.kernel.map(K.index)
df = df.sort_values(["a", "k", "n_components", "test_size"]).drop(columns=["a", "k"]).reset_index(drop=True)
df.to_csv("results/results.csv", index=False)
KN = {"rbf": "RBF", "sigmoid": "Sigmoid", "linear": "Linear", "poly": "Polynomial", "cosine": "Cosine"}
M = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score")]
rows = []
for a in A:
    first = True
    for k in K:
        for nc in sorted(df[df.kernel == k].n_components.unique()):
            g = df[(df.algorithm == a) & (df.kernel == k) & (df.n_components == nc)].set_index("test_size")
            r = {("", "Algorithm"): g.model.iloc[0] if first else "", ("", "Kernel PCA"): KN[k], ("", "nComp"): nc}
            first = False
            for ts in sorted(g.index):
                for m, n in M: r[(f"Test size {ts}", n)] = round(g.loc[ts, m], 4)
            rows.append(r)
main = pd.DataFrame(rows); main.columns = pd.MultiIndex.from_tuples(main.columns)
cols = dict(model="Model Name", kernel="Kernel PCA", n_components="nComp", test_size="Test Size", n_train="Train Samples",
            n_test="Test Samples", accuracy="Accuracy", precision="Precision", recall="Recall", f1="F1 Score")
with pd.ExcelWriter("results/model_performance_comparison.xlsx") as xw:
    main.to_excel(xw, sheet_name="Algorithm by Kernel")
    df[list(cols)].rename(columns=cols).round(4).to_excel(xw, sheet_name="Comparison by Split", index=False)
    best = df.loc[df.groupby("algorithm").accuracy.idxmax()].set_index("algorithm").loc[A]
    best[["model", "kernel", "n_components", "test_size", "accuracy", "precision", "recall", "f1"]].round(4).to_excel(xw, sheet_name="Best per Algorithm")
    for a in A:
        g = df[df.algorithm == a].copy(); g["row"] = g.kernel.map(KN) + " (nComp=" + g.n_components.astype(str) + ")"
        for m in ("accuracy", "f1"):
            p = g.pivot(index="row", columns="test_size", values=m).round(4); p.columns = [f"Test {c}" for c in p.columns]
            p.to_excel(xw, sheet_name=f"{a} {m}")
print(df.pivot_table(index=["algorithm", "kernel"], columns="test_size", values="accuracy", aggfunc="max").round(3))
print(best[["kernel", "n_components", "test_size", "accuracy"]])
