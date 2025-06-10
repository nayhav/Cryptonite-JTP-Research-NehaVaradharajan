TASK 1 NOTES

Source: UCI Heart Disease Dataset (Cleveland) https://archive.ics.uci.edu/dataset/45/heart+disease

Target Column: num (transformed into binary: 0 = no disease, 1 = disease)
Total Samples: 303
Features: 13 numeric/categorical clinical features
Missing values:
ca: 4 missing
thal: 2 missing( These rows were dropped during preprocessing )

This project focused on predicting the presence of heart disease using the UCI Cleveland dataset. The objective was to evaluate and compare the performance of different machine learning models on the dataset and identify the most effective approach for this classification problem.

The dataset used was processed.cleveland.data, which contains patient medical attributes and a target variable (num) indicating the presence of heart disease. During preprocessing, missing values (denoted by '?') were identified and replaced using median imputation. Feature scaling was applied using standardization with StandardScaler to ensure that all input features were on a comparable scale. The target variable was binarized so that 0 indicates absence of disease and any non-zero value is treated as 1, indicating the presence of heart disease.

The dataset was split into training and testing sets using an 80-20 ratio with stratification to maintain class distribution. A fixed random seed was used to ensure reproducibility across experiments.

Three machine learning models were implemented and evaluated: Logistic Regression, Random Forest Classifier, and Support Vector Classifier (SVC). For each model, hyperparameter tuning was performed using GridSearchCV with five-fold cross-validation, optimizing for the ROC AUC score.

The Logistic Regression model performed well with an accuracy of 0.852 and a ROC AUC of 0.958. It achieved good precision and recall, especially for both classes, indicating balanced performance. However, it slightly underperformed in comparison to the other two models.

The Random Forest Classifier showed slightly better performance with an accuracy of 0.885 and a ROC AUC of 0.959. It also provided useful feature importance values, offering better model interpretability. It was able to capture nonlinear relationships and interactions between features more effectively.

The SVC model achieved the same accuracy as the Random Forest (0.885) and a slightly lower ROC AUC of 0.944. It also produced high precision and recall, especially for detecting cases with heart disease. Although SVC does not naturally provide feature importances, it performed comparably to Random Forest in predictive power.

Based on the evaluation, the Random Forest model performed best overall. It had the highest ROC AUC and matched the best accuracy score. Its ability to handle feature interactions, resistance to overfitting through parameter tuning, and interpretability through feature importances make it the preferred choice among the three tested models.

The entire workflow, including data cleaning, preprocessing, model training, evaluation, and visualization, was implemented in Python using libraries such as pandas, scikit-learn, seaborn, and matplotlib.
