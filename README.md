# Assignment 15 — Core Algorithms, Metrics & Model Behavior

## 📊 Dataset
This assignment uses a synthetic dataset generated within the script.

Columns:
- math score
- reading score
- writing score
- gender

The dataset is saved as:
dataset.csv

---

## ⚙️ Tasks Implemented

### PART 1 — Regression
- Linear Regression model trained
- Predictions made
- Actual vs Predicted plotted

### PART 2 — Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

### PART 3 — Classification
- Logistic Regression
- Naive Bayes (GaussianNB)
- K-Nearest Neighbors (KNN with k=3,5,7)

---

### PART 4 — Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

### PART 5 — Model Behavior

#### Overfitting vs Underfitting
- Simple model (underfitting)
- Complex model (overfitting)
- Compared training and testing accuracy

---

### Bias & Variance

- Bias: Error due to overly simple model
- Variance: Error due to overly complex model

Relation:
- High bias → Underfitting
- High variance → Overfitting

How to reduce overfitting:
- Use simpler models
- Add more data
- Apply regularization

---

## 🛠️ Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib

---

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python setup_all.py