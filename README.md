# Heart Disease Detection — Voting Classifier with K-Fold Cross Validation

An ensemble machine learning model for heart disease prediction using a Soft Voting Classifier combining **Random Forest**, **Gradient Boosting**, and **Logistic Regression**, evaluated with **10-Fold Cross Validation**.

## Model Performance (10-Fold CV Average)

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | 93.64%    |
| Sensitivity | 89.93%    |
| Specificity | 97.52%    |
| Precision   | 97.84%    |
| Recall      | 89.93%    |
| F1 Score    | 93.67%    |
| Kappa Score | 87.19%    |
| AUC         | 98.97%    |

## How It Works

- K=1 cross-validation is mathematically impossible (zero training data), so minimum K=2 is used
- Soft voting averages predicted probabilities from all 3 classifiers
- Prediction threshold: **0.64** — if average ensemble probability ≥ 0.64, class = 1 (Heart Disease)

## Dataset
[Google Sheets Dataset](https://docs.google.com/spreadsheets/d/1GwUew6wtbX0OPm9vGSfs4hSu1spH2wDi0MBiLwcGulY/edit?usp=sharing)

## Code
[Google Colab Notebook](https://colab.research.google.com/drive/1F-QNBrrR6JFIECzQ_kkn2GrgsR_pUvGc?usp=sharing)

## Tech Stack
`Python` `Scikit-learn` `Pandas` `NumPy` `Ensemble Learning` `K-Fold Cross Validation`
