# ACN PROJECT V3 - Kernel PCA + Classifiers for Network Intrusion Detection (CIC-IDS2017)

A controlled comparison of five classifiers on CIC-IDS2017 flow data after **Kernel PCA** dimensionality reduction.
Grid: **3 test sizes x 5 Kernel-PCA kernels x 5 algorithms = 75 experiments** (75 scripts + 75 JPGs, 15 + 15 per algorithm folder), one script and one result image per experiment.

## Experimental design

| Factor | Values |
|---|---|
| Test size | 0.2, 0.4, 0.6 |
| Kernel PCA kernel (10 components each) | Sigmoid, Cosine, Linear, RBF, Polynomial |
| Classifiers | Logistic Regression, KNN (k=5), Support Vector Machine (RBF, C=1), Decision Tree (entropy), Random Forest (100 trees) |

- **Data:** CIC-IDS2017 Tuesday + Wednesday + Thursday-morning CSVs (1,308,978 flows, 11 classes: BENIGN, DoS Hulk / GoldenEye / slowloris / Slowhttptest, FTP-Patator, SSH-Patator, three web attacks, Heartbleed).
- **Cleaning:** strip column names, coerce to numeric, replace inf with NaN, mean-impute, label-encode.
- **Sample:** Kernel PCA is O(n^2) in memory/time, so the experiments use a **1000-row stratified sample** (>= 10 rows per class so rare attacks survive stratified splitting; 718 of 1000 rows are BENIGN).
- **Pipeline per run:** stratified split (`random_state=42`) -> `StandardScaler` (fit on train) -> `KernelPCA` (fit on train, default gamma) -> classifier -> weighted precision / recall / F1 + confusion matrix.
- **Note:** variant 3 of the project: five Kernel-PCA kernels per algorithm (the notes' 15 x 5 = 75 runs). The number of components is fixed at 10 (`N_COMP` in `make_scripts.py`); the notes do not state it, so change it there if you need another value.

## Results (accuracy / weighted F1)

| Algorithm | Kernel PCA | Test 0.2 Acc / F1 | Test 0.4 Acc / F1 | Test 0.6 Acc / F1 |
|---|---|---|---|---|
| Logistic Regression | Sigmoid | 0.8150 / 0.7593 | 0.8200 / 0.7656 | 0.8333 / 0.7789 |
|  | Cosine | 0.8500 / 0.7977 | 0.8525 / 0.8002 | 0.8567 / 0.8042 |
|  | Linear | 0.8500 / 0.8118 | 0.8600 / 0.8249 | 0.8583 / 0.8229 |
|  | RBF | 0.8350 / 0.7815 | 0.8375 / 0.7844 | 0.8350 / 0.7815 |
|  | Polynomial | 0.8100 / 0.7565 | 0.8275 / 0.7816 | 0.8267 / 0.7839 |
| KNN (k=5) | Sigmoid | 0.8900 / 0.8854 | 0.8925 / 0.8840 | 0.8900 / 0.8836 |
|  | Cosine | 0.8800 / 0.8820 | 0.8900 / 0.8853 | 0.8767 / 0.8730 |
|  | Linear | 0.8650 / 0.8613 | 0.8850 / 0.8793 | 0.8867 / 0.8795 |
|  | RBF | 0.8600 / 0.8592 | 0.8900 / 0.8808 | 0.8700 / 0.8646 |
|  | Polynomial | 0.8500 / 0.8495 | 0.8750 / 0.8668 | 0.8633 / 0.8581 |
| Support Vector Machine (RBF) | Sigmoid | 0.8750 / 0.8368 | 0.8775 / 0.8371 | 0.8800 / 0.8410 |
|  | Cosine | 0.8800 / 0.8435 | 0.8625 / 0.8282 | 0.8650 / 0.8196 |
|  | Linear | 0.8500 / 0.8063 | 0.8475 / 0.8064 | 0.8517 / 0.8075 |
|  | RBF | 0.8550 / 0.8026 | 0.8600 / 0.8106 | 0.8583 / 0.8080 |
|  | Polynomial | 0.8100 / 0.7549 | 0.8225 / 0.7709 | 0.8267 / 0.7753 |
| Decision Tree (entropy) | Sigmoid | 0.9000 / 0.9051 | 0.8975 / 0.9005 | 0.8700 / 0.8706 |
|  | Cosine | 0.8800 / 0.8804 | 0.8925 / 0.8938 | 0.8733 / 0.8752 |
|  | Linear | 0.9000 / 0.9062 | 0.8775 / 0.8765 | 0.9083 / 0.9106 |
|  | RBF | 0.8450 / 0.8614 | 0.8675 / 0.8763 | 0.8833 / 0.8829 |
|  | Polynomial | 0.8850 / 0.8888 | 0.9050 / 0.9044 | 0.9000 / 0.9011 |
| Random Forest (100 trees) | Sigmoid | 0.9100 / 0.9030 | 0.9275 / 0.9216 | 0.9317 / 0.9268 |
|  | Cosine | 0.9150 / 0.9143 | 0.9200 / 0.9150 | 0.9133 / 0.9069 |
|  | Linear | 0.9150 / 0.9111 | 0.9275 / 0.9239 | 0.9300 / 0.9237 |
|  | RBF | 0.8900 / 0.8858 | 0.9075 / 0.9002 | 0.9183 / 0.9112 |
|  | Polynomial | 0.9200 / 0.9147 | 0.9250 / 0.9188 | 0.9233 / 0.9172 |

Best run: **Random Forest (100 trees), sigmoid Kernel PCA, test size 0.6 - accuracy 0.9317**. Mean accuracy over all 15 runs per algorithm: Logistic Regression 0.838, KNN 0.878, Support Vector Machine 0.855, Decision Tree 0.886, Random Forest 0.918.

Full tables (all four metrics, per-algorithm grids) are in `results/model_performance_comparison.xlsx`; raw rows in `results/results.csv`.
Each script saves one JPG beside it with its console output, metric chart and confusion matrix, e.g. `random_forest/Test0.6_linear_RF.jpg`.

## Limitations - please read

- **Small sample, single split.** 1000 rows and one split per run: rare attack classes have only 2-6 test rows, so differences of a point or two are noise. No cross-validation or significance tests.
- **Accuracy is inflated by class imbalance.** Predicting BENIGN for everything scores 71.8% on this sample; compare weighted F1 and the confusion matrices.
- **Not comparable to full-data results** reported in the literature (typically 98-99%).
- Uses 3 of the 8 CIC-IDS2017 files; no hyper-parameter tuning; no no-reduction baseline.

## Reproduce

```bash
pip install -r requirements.txt

# 1. Download CIC-IDS2017 "MachineLearningCSV.zip" from the Canadian Institute for Cybersecurity
#    (https://www.unb.ca/cic/datasets/ids-2017.html) and unzip so the CSVs are in ./MachineLearningCVE/
python prepare_data.py MachineLearningCVE   # builds data/sample.npz (a prebuilt copy is included)

python make_scripts.py                      # (re)generates the 75 scripts
python run_all.py                           # runs all 75 and builds results/ workbook + CSV
python logistic/Test0.2_sigmoid_LR.py      # or run a single experiment
```

## Run in Spyder / VS Code / PyCharm / Jupyter

Open any script under `<algorithm>/` and press Run (F5 in Spyder). No arguments and no working-directory setup are needed: it finds `data/sample.npz` itself,
prints the confusion matrix, metrics and classification report to the console, shows the result image, and saves it as `<name>.jpg` next to the script. The JPG contains the console output plus the metric chart and confusion-matrix heatmap.
`run_all.py` runs all 75 without opening chart windows.

## Repository layout

```
prepare_data.py     load + clean + stratified 1000-row sample -> data/sample.npz
make_scripts.py     generates the 75 experiment scripts
run_all.py          runs everything, writes results/
data/sample.npz     cached sample (so scripts run without the raw CSVs)
<algo>/             Test<size>_<kernel>_<ALG>.py  +  matching .jpg (console output + charts)
results/            results.csv, model_performance_comparison.xlsx, json/ (one metrics file per run)
```

## Dataset citation

Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018.
