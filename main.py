import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("data/credit_data.csv")

print("\n========== DATASET PREVIEW ==========\n")
print(data.head())

# =========================
# FEATURES & TARGET
# =========================

X = data.drop("Creditworthy", axis=1)
y = data["Creditworthy"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = {}

# =========================
# TRAINING & EVALUATION
# =========================

for name, model in models.items():

    print(f"\n========== {name} ==========\n")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Probability scores
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Store results
    results[name] = accuracy

    # Print metrics
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # =========================
    # CONFUSION MATRIX
    # =========================

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"outputs/{name}_confusion_matrix.png")
    plt.close()

    # =========================
    # ROC CURVE
    # =========================

    RocCurveDisplay.from_predictions(y_test, y_prob)

    plt.title(f"{name} - ROC Curve")
    plt.savefig(f"outputs/{name}_roc_curve.png")
    plt.close()

# =========================
# FEATURE IMPORTANCE
# =========================

rf_model = models["Random Forest"]

importances = rf_model.feature_importances_
features = X.columns

feature_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

plt.barh(feature_df["Feature"], feature_df["Importance"])

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")

plt.savefig("outputs/feature_importance.png")
plt.close()

# =========================
# BEST MODEL
# =========================

best_model = max(results, key=results.get)

print("\n========================")
print(f"BEST MODEL: {best_model}")
print("========================")

# =========================
# SAVE RESULTS
# =========================

with open("outputs/model_results.txt", "w") as file:

    file.write("CREDIT SCORING MODEL RESULTS\n\n")

    for model_name, accuracy in results.items():
        file.write(f"{model_name}: {accuracy:.4f}\n")

print("\nResults saved successfully.")
