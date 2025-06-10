import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, roc_auc_score,
                             accuracy_score)

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_and_preprocess_data(filepath):
    # Define column names based on UCI dataset documentation
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    # Load dataset with column names, treating '?' as NaN
    df = pd.read_csv(filepath, names=columns, na_values='?')
    
    # Convert target to binary: 0 = no disease, 1 = disease
    df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    print("----- Dataset Head -----")
    print(df.head())
    print("\n----- Missing Values per Column -----")
    print(df.isnull().sum())

    target = 'num'

    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X = df.drop(columns=[target])
    y = df[target]

    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    return X_scaled, y

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_test, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    return auc_score

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    print(f" Hyperparameter tuning for {model.__class__.__name__} ...")
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params for {model.__class__.__name__}: {grid.best_params_}")
    return grid.best_estimator_

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model_name = model.__class__.__name__
    print(f"\n Training and evaluating: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    print(f"\n Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

    auc_score = plot_roc_curve(y_test, y_proba, model_name)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, auc_score, model

def plot_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10,6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
    else:
        print("Model does not provide feature importances.")

def main():
    filepath = "processed.cleveland.data"  
    X, y = load_and_preprocess_data(filepath)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Define models and parameter grids for tuning
    models_and_params = [
        (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        ),
        (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
        ),
        (
            SVC(random_state=RANDOM_STATE, probability=True),
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        )
    ]

    results = []

    for model, params in models_and_params:
        best_model = hyperparameter_tuning(model, params, X_train, y_train)
        accuracy, auc, trained_model = evaluate_model(best_model, X_train, y_train, X_test, y_test)
        results.append({
            'Model': model.__class__.__name__,
            'Accuracy': accuracy,
            'ROC_AUC': auc,
            'TrainedModel': trained_model
        })

    # Feature importance for Random Forest
    rf_model = next(res['TrainedModel'] for res in results if res['Model'] == "RandomForestClassifier")
    print("\n Feature Importance for Random Forest:")
    plot_feature_importances(rf_model, X.columns)

    # Summary Table
    print("\n Performance Summary")
    print("-" * 40)
    print(f"{'Model':<25} {'Accuracy':<10} {'ROC AUC':<10}")
    print("-" * 40)
    for res in results:
        print(f"{res['Model']:<25} {res['Accuracy']:<10.3f} {res['ROC_AUC']:<10.3f}")
    print("-" * 40)
    print("Models Implemented")

if __name__ == "__main__":
    main()
