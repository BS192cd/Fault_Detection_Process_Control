# Tennessee Eastman Fault Detection — PCA & LightGBM

This repository contains a Jupyter notebook that applies Principal Component Analysis (PCA) and LightGBM (LGB) ensemble regressors to the Tennessee Eastman Process (TEP) benchmark for fault detection. The notebook evaluates preprocessing strategies (lagged variables and moving-average noise filtering) and compares the detection performance of PCA-based SPE (Squared Prediction Error) and a LightGBM ensemble approach.

**Notebook**: `tennessee-eastman-fault-detection-with-pca-and-lgb.ipynb`

## Project Overview
- **Goal:** Detect process faults in the Tennessee Eastman simulation using unsupervised (PCA) and supervised/ensemble regression (LGB) anomaly-detection strategies.
- **Key techniques:** PCA (SPE-based detection), ensemble of regressors using LightGBM to predict each feature from the rest (Averkiev-style), dynamical PCA (lagged variables), and moving-average noise filtering.
- **Inputs:** TEP RData files for fault-free training/validation/testing and faulty test sets (loaded with `pyreadr`).
- **Outputs:** SPE time series for each method, detection limits (percentile-based), fault detection rates (FDR) per fault, and plots comparing methods.

## Notebook Structure (high-level)
1. Environment and required libraries installation/imports.
2. Definition of helper classes and functions:
   - `ModelPCA`: PCA training/test and SPE computation.
   - `ModelEnsembleRegressors`: train separate regressors (LightGBM) for each feature and compute SPE from predictions.
   - `apply_lag`: construct lagged features for DPCA.
   - `filter_noise_ma`: moving-average noise filter.
3. Data loading using `pyreadr` (paths in the notebook assume Kaggle dataset locations).
4. Exploratory plots: training and faulty traces, correlation dendrogram.
5. Training & validation: compute SPE on training/validation/test sets and determine detection thresholds.
6. Testing on faulty sets: compute fault detection rates across 20 fault scenarios.
7. Experiments: add lag variables (DPCA) and apply moving-average filters with several window sizes for PCA and LGB.
8. Conclusions summarizing which methods perform best globally and per-fault.

## Requirements
Recommended Python packages (the notebook installs `pyreadr==0.3.3` inline):

```
numpy
pandas
matplotlib
scipy
scikit-learn
lightgbm
pyreadr==0.3.3
statsmodels
jupyter
```

You can create a virtual environment and install the requirements:

```powershell
# from project root
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy pandas matplotlib scipy scikit-learn lightgbm pyreadr==0.3.3 statsmodels jupyter
```

## How to run
1. Ensure the TEP dataset RData files are available. The notebook expects paths similar to Kaggle dataset locations; update the paths in the notebook cells if your files are in a different location.
2. Start Jupyter and open the notebook from the repository root:

```powershell
jupyter notebook
# then open: Fault_Detection_Process_Control/tennessee-eastman-fault-detection-with-pca-and-lgb.ipynb
```

3. Run the cells sequentially. Some long-running cells include training an ensemble of LightGBM models (one per feature) and plotting; allow time for model fitting.

## Notes & Reproducibility
- The notebook uses percentile (99.99th) of validation SPE to set detection thresholds — you can adjust this if you want different operating points.
- Data paths in the notebook point to Kaggle input directories; change them to local paths where you extracted the TEP RData files.
- For reproducible results, consider setting random seeds for LightGBM and NumPy where models are instantiated and trained.

## Results (summary from notebook)
- PCA (SPE-based) generally outperforms the LightGBM ensemble in a global sense, but LGB can reach comparable performance when noise filtering (moving average) is applied.
- Adding lagged variables (DPCA) helps in some faults (notably faults 11 and 14) but does not consistently increase global performance.

## File structure
- `tennessee-eastman-fault-detection-with-pca-and-lgb.ipynb` — main analysis notebook.

## License & Contact
This repository does not include a license file. If you plan to reuse code, please add an appropriate license or contact the repository owner for permission.

If you want, I can:
- add a `requirements.txt` or `environment.yml` file,
- update notebook data paths to relative local paths and add small helper scripts to download/prepare the dataset,
- or commit the README changes to a branch and open a PR (if you want me to do that here).

---
Generated summary README for the notebook in this repository.
Process control
