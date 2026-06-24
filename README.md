# Explainable Detection of Online Sexism (EDOS) — Task A

## Project Overview
This project tackles **Task A (Binary Sexism Detection)** from SemEval-2023 Task 10 (EDOS):
given a social media post, predict whether it is **sexist** or **not sexist**. We work on a
5k-instance subset of the original dataset and build a fully classical, interpretable NLP
pipeline, comparing several feature representations and models to maximize weighted F1 on the
held-out test set.

## Pipeline
1. **Data Preprocessing** — lowercasing, removal of URLs, `@mentions`, `[user]`/`[url]`
   placeholder tokens, and whitespace normalization. The dataset's predefined `train`/`test`
   split is respected to avoid any test-set leakage.
2. **Feature Engineering (2 methods)**
   - **Method 1 — Word TF-IDF:** word n-grams (1–3), English stopwords removed, `min_df=2`.
   - **Method 2 — Character TF-IDF:** char n-grams (3–5), `min_df=3`, to capture subword
     patterns in informal/misspelled text.
   - **Combined:** horizontal stacking of both representations.
3. **Modeling (3 models)** — each trained on both feature methods and on the combined matrix:
   - Logistic Regression (`liblinear`, `class_weight="balanced"`, `C=0.5`)
   - Linear SVM (`LinearSVC`)
   - XGBoost (`tree_method="hist"`, tuned `n_estimators`, `learning_rate`, `max_depth`)
4. **Evaluation** — `classification_report` from scikit-learn, reporting Precision, Recall and
   F1 (per class and weighted average) on the **1,086-instance test set**.

## Key Results
- Best performance: **XGBoost on the combined (word + char TF-IDF) features**, reaching
  ~**0.82 accuracy** and **~0.81 weighted F1** on the test set.
- Classical TF-IDF + gradient boosting proved to be a strong, transparent baseline, with
  character-level features giving a consistent boost over word-level features alone.

## Technologies Used
- **Language:** Python
- **Libraries:** scikit-learn, XGBoost, pandas, NumPy
- **Environment:** Jupyter Notebook / Google Colab
- **Version Control:** Git / GitHub

## Authors
* Rena Wang  
* Marco Gonzalez
