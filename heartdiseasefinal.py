#Name: Neha Varadharajan
#Branch: Computer Science and Engineering (AI&ML)
#Models for Cleveland Heart Disease Prediction
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

# Setting a seed value to ensure results are consistent each time we run the code
RANDOM_STATE = 42

# Step 1: Load and preprocess the data
def load_and_preprocess_data(filepath):
    # Define the column names based on the dataset's documentation
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    # Read the CSV file, marking '?' as missing values
    df = pd.read_csv(filepath, names=columns, na_values='?')

    # Convert the target variable into binary (0 = no disease, 1 = has disease)
    df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    # Display first few rows and check for missing values
    print("----- Dataset Head -----")
    print(df.head())
    print("\n----- Missing Values per Column -----")
    print(df.isnull().sum())

    target = 'num'  # This is the column we want to predict

    # Use median to fill missing values
    imputer = SimpleImputer(strategy='median')
    X = df.drop(columns=[target])
    y = df[target]
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Standardize the data to bring all features to the same scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    return X_scaled, y

# Function to plot confusion matrix for visualizing performance
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot ROC curve and calculate AUC score
def plot_roc_curve(y_test, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    return auc_score

# Function to tune hyperparameters using GridSearchCV
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    print(f"\nHyperparameter tuning for {model.__class__.__name__} ...")
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params for {model.__class__.__name__}: {grid.best_params_}")
    return grid.best_estimator_

# Function to evaluate model performance using various metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model_name = model.__class__.__name__
    print(f"\nTraining and evaluating: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get predicted probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For SVM which uses decision_function
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    # Print classification report
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

    # Plot ROC curve
    auc_score = plot_roc_curve(y_test, y_proba, model_name)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, auc_score, model

# Function to plot the importance of each feature (only for tree-based models)
def plot_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
    else:
        print("Model does not provide feature importances.")

# Main workflow
def main():
    # Load and prepare the dataset
    filepath = "processed.cleveland.data"
    X, y = load_and_preprocess_data(filepath)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Define models and their hyperparameter grids
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
        ),
        (
            XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
            {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
        )
    ]

    results = []

    # Loop over each model, tune hyperparameters, train and evaluate
    for model, params in models_and_params:
        best_model = hyperparameter_tuning(model, params, X_train, y_train)
        accuracy, auc, trained_model = evaluate_model(best_model, X_train, y_train, X_test, y_test)
        results.append({
            'Model': model.__class__.__name__,
            'Accuracy': accuracy,
            'ROC_AUC': auc,
            'TrainedModel': trained_model
        })

    # Plot feature importance for Random Forest
    rf_model = next(res['TrainedModel'] for res in results if res['Model'] == "RandomForestClassifier")
    print("\nFeature Importance for Random Forest:")
    plot_feature_importances(rf_model, X.columns)

    # Plot feature importance for XGBoost
    xgb_model = next(res['TrainedModel'] for res in results if res['Model'] == "XGBClassifier")
    print("\nFeature Importance for XGBoost:")
    plot_feature_importances(xgb_model, X.columns)

    # Display performance summary
    print("\nPerformance Summary")
    print("-" * 40)
    print(f"{'Model':<25} {'Accuracy':<10} {'ROC AUC':<10}")
    print("-" * 40)
    for res in results:
        print(f"{res['Model']:<25} {res['Accuracy']:<10.3f} {res['ROC_AUC']:<10.3f}")
    print("-" * 40)
    print("Models Implemented")

# Run the script
if __name__ == "__main__":
    main()
