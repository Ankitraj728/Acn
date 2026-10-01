"""Step 1: load Tue/Wed/Thu-morning CICIDS2017, clean (as in ACFROG1), draw a
stratified N-row sample and cache it. Kernel PCA is O(n^2) so the full 1.3M rows
cannot be used; N_SAMPLES=1000 matches the handwritten plan (200/400/600 test rows)."""
import sys, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else "MachineLearningCVE"
N_SAMPLES = 1000
MIN_PER_CLASS = 10   # keeps rare attack classes present so stratified splits work
SEED = 0

files = ["Tuesday-WorkingHours.pcap_ISCX.csv",
         "Wednesday-workingHours.pcap_ISCX.csv",
         "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"]
dataset = pd.concat([pd.read_csv(f"{CSV_DIR}/{f}", low_memory=True) for f in files],
                    ignore_index=True)
dataset.columns = dataset.columns.str.strip()
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].str.strip()
print("rows:", len(dataset)); print(y.value_counts())

X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(X)
le = LabelEncoder(); y_enc = le.fit_transform(y)

# stratified sample: >= MIN_PER_CLASS per class, rest proportional
rng = np.random.RandomState(SEED)
classes, counts = np.unique(y_enc, return_counts=True)
alloc = {c: min(n, MIN_PER_CLASS) for c, n in zip(classes, counts)}
rest = N_SAMPLES - sum(alloc.values())
prop = counts / counts.sum()
for c, n, p in zip(classes, counts, prop):
    alloc[c] = min(n, alloc[c] + int(round(rest * p)))
alloc[classes[np.argmax(counts)]] += N_SAMPLES - sum(alloc.values())  # round-off top-up
idx = np.concatenate([rng.choice(np.where(y_enc == c)[0], alloc[c], replace=False) for c in classes])
rng.shuffle(idx)
print("sample size:", len(idx))
np.savez_compressed("data/sample.npz", X=X[idx], y=y_enc[idx], classes=le.classes_)
print({le.classes_[c]: alloc[c] for c in classes})
