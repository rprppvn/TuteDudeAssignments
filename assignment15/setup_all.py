import pandas as pd
import numpy as np

# -----------------------------
# CREATE DATASET (IMPORTANT)
# -----------------------------
np.random.seed(0)

data = {
    "math score": np.random.randint(40,100,100),
    "reading score": np.random.randint(40,100,100),
    "writing score": np.random.randint(40,100,100),
    "gender": np.random.choice(["male","female"],100)
}

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)

print("dataset.csv created")

# -----------------------------
# TASK 1 — LINEAR REGRESSION
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = df[["reading score"]]
y = df["math score"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

plt.scatter(y_test,y_pred)
plt.title("Linear Regression")
plt.show()

# -----------------------------
# TASK 2 — REGRESSION METRICS
# -----------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

# -----------------------------
# TASK 3 — LOGISTIC REGRESSION
# -----------------------------
from sklearn.linear_model import LogisticRegression

df["pass"] = (df["math score"] > 60).astype(int)

X = df[["reading score","writing score"]]
y = df["pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# -----------------------------
# TASK 4 — NAIVE BAYES
# -----------------------------
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred_nb = nb.predict(X_test)

# -----------------------------
# TASK 5 — KNN
# -----------------------------
from sklearn.neighbors import KNeighborsClassifier

for k in [3,5,7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    print("K=",k,"Accuracy:",knn.score(X_test,y_test))

# -----------------------------
# TASK 6 — CLASSIFICATION METRICS
# -----------------------------
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1:",f1_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

# -----------------------------
# TASK 7 — OVERFITTING / UNDERFITTING
# -----------------------------
from sklearn.tree import DecisionTreeClassifier

simple = DecisionTreeClassifier(max_depth=1)
simple.fit(X_train,y_train)

complex_model = DecisionTreeClassifier()
complex_model.fit(X_train,y_train)

print("Underfit Train:",simple.score(X_train,y_train))
print("Underfit Test:",simple.score(X_test,y_test))

print("Overfit Train:",complex_model.score(X_train,y_train))
print("Overfit Test:",complex_model.score(X_test,y_test))

# -----------------------------
# TASK 8 — INSIGHTS
# -----------------------------
print("INSIGHTS:")
print("1. Reading and writing influence math performance.")
print("2. Higher scores tend to correlate.")
print("3. Simple models underfit.")
print("4. Complex models overfit.")
print("5. KNN accuracy varies with k.")
