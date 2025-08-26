
# Importing necessary libraries for data manipulation, visualization, and modeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

# Importing preprocessing and modeling tools from sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, roc_auc_score,
                             accuracy_score)

# Importing XGBoost model
from xgboost import XGBClassifier

# Setting a seed value to ensure results are consistent
SEED = 42

# Load and preprocess the data
def load_data(path):
    cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    df = pd.read_csv(path, names=cols, na_values='?')
    df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    print("----- Dataset Head -----")
    print(df.head())
    print("\n----- Missing Values per Column -----")
    print(df.isnull().sum())

    target = 'num'
    imputer = SimpleImputer(strategy='median')
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

# Function to plot confusion matrix
def plot_cm(cm, name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot ROC curve
def plot_roc(y_true, y_prob, name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    return auc

# Function for hyperparameter tuning
def tune(model, params, X_train, y_train):
    print(f"\nHyperparameter tuning for {model.__class__.__name__} ...")
    grid = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params for {model.__class__.__name__}: {grid.best_params_}")
    return grid.best_estimator_

# Function to evaluate model
def evaluate(model, X_train, y_train, X_test, y_test):
    name = model.__class__.__name__
    print(f"\nTraining and evaluating: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, name)

    auc = plot_roc(y_test, y_prob, name)
    acc = accuracy_score(y_test, y_pred)

    return acc, auc, model

# Function to plot feature importances
def plot_importance(model, features):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        idx = np.argsort(importance)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance[idx], y=np.array(features)[idx])
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
    else:
        print("Model does not provide feature importances.")

# Main workflow
def main():
    path = "processed.cleveland.data"
    X, y = load_data(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    configs = [
        (
            LogisticRegression(random_state=SEED, max_iter=1000),
            {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        ),
        (
            RandomForestClassifier(random_state=SEED),
            {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
        ),
        (
            SVC(random_state=SEED, probability=True),
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        ),
        (
            XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss'),
            {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
        )
    ]

    results = []

    for model, params in configs:
        best = tune(model, params, X_train, y_train)
        acc, auc, trained = evaluate(best, X_train, y_train, X_test, y_test)
        results.append({
            'Model': model.__class__.__name__,
            'Accuracy': acc,
            'ROC_AUC': auc,
            'Trained': trained
        })

    rf = next(r['Trained'] for r in results if r['Model'] == "RandomForestClassifier")
    print("\nFeature Importance for Random Forest:")
    plot_importance(rf, X.columns)

    xgb = next(r['Trained'] for r in results if r['Model'] == "XGBClassifier")
    print("\nFeature Importance for XGBoost:")
    plot_importance(xgb, X.columns)

    print("\nPerformance Summary")
    print("-" * 40)
    print(f"{'Model':<25} {'Accuracy':<10} {'ROC AUC':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['Model']:<25} {r['Accuracy']:<10.3f} {r['ROC_AUC']:<10.3f}")
    print("-" * 40)
    print("Models Implemented")


if __name__ == "__main__":
    main()
