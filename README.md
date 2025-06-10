TASK 1 NOTES

Source: UCI Heart Disease Dataset (Cleveland) https://archive.ics.uci.edu/dataset/45/heart+disease

Target Column: num (transformed into binary: 0 = no disease, 1 = disease)
Total Samples: 303
Features: 13 numeric/categorical clinical features
Missing values:
ca: 4 missing
thal: 2 missing( These rows were dropped during preprocessing )

The objective of this project was to build a reliable machine learning model to predict the presence of heart disease using the UCI Heart Disease dataset. The dataset was first preprocessed by assigning appropriate column names and handling missing values, which were denoted using question marks. These missing values were imputed using the median strategy to preserve the central tendency of each feature without being affected by outliers. After imputing, all numerical features were standardized using z-score normalization via StandardScaler to ensure consistent scale across features, which is critical for models like logistic regression and SVM.

The target variable 'num' was converted to a binary classification problem where 0 indicated no presence of heart disease and 1 indicated presence. The dataset was split into training and testing sets using an 80-20 stratified split to maintain the proportion of target classes in both sets. This helped ensure that the models generalize better and are not biased by class imbalance.

Four machine learning models were implemented and evaluated: logistic regression, random forest classifier, support vector classifier (SVC), and XGBoost classifier. Hyperparameter tuning was carried out using GridSearchCV with 5-fold cross-validation to identify the optimal configuration for each model based on ROC-AUC score. After training, each model was evaluated using multiple performance metrics including accuracy, ROC-AUC score, confusion matrix, and ROC curve plots. For the tree-based models—random forest and XGBoost—feature importances were also visualized to understand which clinical attributes played the most significant role in prediction.

The random forest and SVC models achieved the highest accuracy of 88.5 percent. Among them, random forest had the best ROC-AUC score of 0.959, suggesting not only high predictive power but also good separation between the classes. Logistic regression, while slightly lower in accuracy, performed competitively with an ROC-AUC of 0.958, demonstrating its strength as a baseline model for this type of structured clinical data. XGBoost, often regarded as one of the most powerful classifiers, achieved an accuracy of 85.2 percent and a ROC-AUC of 0.927 in this particular run. While its performance was slightly behind the others, it still proved robust, and its efficiency and flexibility make it worth considering in future iterations, especially with deeper hyperparameter tuning or ensemble strategies.

Overall, the random forest model provided the most balanced and interpretable performance across all evaluation metrics, making it the most suitable model for predicting heart disease in this context. The workflow followed in this project—from data preprocessing and imputation, to model training, evaluation, and interpretation—demonstrates a complete, reproducible, and insightful machine learning pipeline for solving real-world healthcare classification problems.



TASK 2 NOTES

In this project, we applied three distinct machine learning approaches—linear regression, decision tree regression, and KMeans clustering—to analyze the Individual Household Electric Power Consumption dataset. Our primary objective was to explore different modeling techniques for understanding and predicting electricity usage patterns.

The regression models focused on predicting global active power using features such as voltage, global intensity, and energy sub-metering. Both linear regression and decision tree regression yielded remarkably high R² scores of 0.998 and low RMSE values, indicating that the relationship between input features and the target variable was strong and largely linear in nature. The linear regression model slightly outperformed the decision tree in terms of RMSE, suggesting that the underlying data was well-suited to linear modeling. However, the decision tree model remains a valuable alternative due to its interpretability and non-parametric nature, which could become advantageous in the presence of non-linearity or interaction effects not evident in this subset of the data.

For unsupervised learning, we applied KMeans clustering to explore patterns in electricity usage without relying on labeled outputs. By plotting clusters using voltage and global intensity, we observed three distinguishable clusters that may correspond to different usage regimes—such as idle, regular use, and high consumption scenarios. While KMeans does not offer a direct predictive utility like regression models, it adds value by uncovering latent structure in user behavior and consumption patterns, which could be useful for segmenting households or optimizing energy delivery strategies.

Overall, linear regression emerged as the most effective model for precise prediction in this task, thanks to its simplicity, speed, and high accuracy on the given data. Decision trees offered comparable results and serve as a robust backup model, especially in real-world scenarios where linearity cannot be assumed. KMeans clustering provided important exploratory insights, complementing the regression models by identifying patterns that are not immediately obvious through prediction alone. Together, these models demonstrate a well-rounded approach to both predictive and exploratory analysis of time-series energy data.

