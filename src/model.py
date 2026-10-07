import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("../data/students_large.csv")

print("Dataset loaded successfully!")
print(df.head())


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df.drop("result", axis=1)

y = df["result"]

y = y.map({
    "Pass": 1,
    "Fail": 0
})


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. Logistic Regression
# ==========================================

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# ==========================================
# 6. Logistic Regression Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


print("\nLogistic Regression Evaluation:")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")



cv_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

cv_scores = cross_val_score(
    cv_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(cv_scores)

print("Mean CV Accuracy:", cv_scores.mean() * 100)
# # ==========================================
# # 7. Classification Report
# # ==========================================

# print("\nClassification Report:")

# print(
#     classification_report(
#         y_test,
#         y_pred,
#         target_names=["Fail", "Pass"]
#     )
# )


# # ==========================================
# # 8. Confusion Matrix
# # ==========================================

# cm = confusion_matrix(y_test, y_pred)

# print("\nConfusion Matrix:")

# print(cm)


# # ==========================================
# # 9. Confusion Matrix Visualization
# # ==========================================

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     xticklabels=["Fail", "Pass"],
#     yticklabels=["Fail", "Pass"]
# )

# plt.xlabel("Predicted")

# plt.ylabel("Actual")

# plt.title("Logistic Regression - Confusion Matrix")

# plt.show()


# # ==========================================
# # 10. Decision Tree
# # ==========================================

# tree_model = DecisionTreeClassifier(
#     random_state=42
# )

# tree_model.fit(X_train, y_train)

# tree_pred = tree_model.predict(X_test)


# # ==========================================
# # 11. Decision Tree Evaluation
# # ==========================================

# tree_accuracy = accuracy_score(
#     y_test,
#     tree_pred
# )

# tree_precision = precision_score(
#     y_test,
#     tree_pred
# )

# tree_recall = recall_score(
#     y_test,
#     tree_pred
# )

# tree_f1 = f1_score(
#     y_test,
#     tree_pred
# )


# # ==========================================
# # 12. Random Forest
# # ==========================================
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [10, 50, 100, 150, 200]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy"
)


grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

forest_model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

importance = forest_model.feature_importances_

for feature, value in zip(X.columns, importance):
    print(feature, ":", value)

plt.bar(X.columns, importance)

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()    


# # ==========================================
# # 13. Random Forest Evaluation
# # ==========================================

forest_accuracy = accuracy_score(
    y_test,
    forest_pred
)

forest_precision = precision_score(
    y_test,
    forest_pred
)

forest_recall = recall_score(
    y_test,
    forest_pred
)

forest_f1 = f1_score(
    y_test,
    forest_pred
)


# # ==========================================
# # 14. Model Comparison
# # ==========================================

# print("\n========================================")

# print("MODEL COMPARISON")

# print("========================================")


# print("\nLogistic Regression")

# print(f"Accuracy: {accuracy * 100:.2f}%")

# print(f"Precision: {precision * 100:.2f}%")

# print(f"Recall: {recall * 100:.2f}%")

# print(f"F1 Score: {f1 * 100:.2f}%")


# print("\nDecision Tree")

# print(f"Accuracy: {tree_accuracy * 100:.2f}%")

# print(f"Precision: {tree_precision * 100:.2f}%")

# print(f"Recall: {tree_recall * 100:.2f}%")

# print(f"F1 Score: {tree_f1 * 100:.2f}%")


print("\nRandom Forest")

print(f"Accuracy: {forest_accuracy * 100:.2f}%")

print(f"Precision: {forest_precision * 100:.2f}%")

print(f"Recall: {forest_recall * 100:.2f}%")

print(f"F1 Score: {forest_f1 * 100:.2f}%")


# # ==========================================
# # 15. Save Logistic Regression Model
# # ==========================================

# joblib.dump(
#     model,
#     "../model.pkl"
# )

# joblib.dump(
#     scaler,
#     "../scaler.pkl"
# )

# print("\nModel and scaler saved successfully!")