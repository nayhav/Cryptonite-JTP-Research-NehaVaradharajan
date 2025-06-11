TASK 1 NOTES

Source: UCI Heart Disease Dataset (Cleveland) https://archive.ics.uci.edu/dataset/45/heart+disease

This machine learning project aimed to build and compare multiple classification models to predict the presence of heart disease using the Cleveland dataset. The target variable, "num", was converted into a binary outcome: 0 for absence of disease and 1 for presence. After handling missing values and scaling the features, four models were implemented and evaluated: Logistic Regression, Random Forest, Support Vector Classifier (SVC), and XGBoost.

Among these models, the Random Forest Classifier emerged as the best performer. It achieved the highest accuracy of 88.5% and the highest ROC AUC score of 0.959. This indicates that it not only made the most correct predictions but also had excellent ability to distinguish between the positive and negative classes across all threshold values. Random Forest's strength lies in its ensemble nature, where multiple decision trees contribute to a more robust and generalized prediction.

The Support Vector Classifier also performed strongly, matching the Random Forest in accuracy with 88.5% but slightly lower in ROC AUC at 0.944. This suggests that while it was equally accurate overall, its probability estimates were not as well calibrated for classification thresholds. Nevertheless, its high precision and recall scores for both classes indicate it is a strong model, especially for problems where margin-based separation is effective.

Logistic Regression and XGBoost both achieved slightly lower accuracy scores of 85.2%. However, Logistic Regression had an impressive ROC AUC of 0.958, suggesting that it performed well in ranking predictions even if some classifications were incorrect. XGBoost, despite its reputation for high performance in many scenarios, lagged slightly behind here with a ROC AUC of 0.927. This might be due to the relatively small size of the dataset or the fact that the optimal hyperparameters for XGBoost were more difficult to pinpoint given the tuning grid.

In summary, Random Forest was the most effective model in this context due to its combination of high accuracy and strong ROC AUC score. It handled the feature interactions and potential non-linearities in the data better than the other models. However, all models performed reasonably well, and the consistent results across different approaches suggest that the dataset is well-suited for binary classification with the applied preprocessing steps. 



TASK 2 NOTES

In this project, we applied three distinct machine learning approaches—linear regression, decision tree regression, and KMeans clustering—to analyze the Individual Household Electric Power Consumption dataset. Our primary objective was to explore different modeling techniques for understanding and predicting electricity usage patterns.

The regression models focused on predicting global active power using features such as voltage, global intensity, and energy sub-metering. Both linear regression and decision tree regression yielded remarkably high R² scores of 0.998 and low RMSE values, indicating that the relationship between input features and the target variable was strong and largely linear in nature. The linear regression model slightly outperformed the decision tree in terms of RMSE, suggesting that the underlying data was well-suited to linear modeling. However, the decision tree model remains a valuable alternative due to its interpretability and non-parametric nature, which could become advantageous in the presence of non-linearity or interaction effects not evident in this subset of the data.

For unsupervised learning, we applied KMeans clustering to explore patterns in electricity usage without relying on labeled outputs. By plotting clusters using voltage and global intensity, we observed three distinguishable clusters that may correspond to different usage regimes—such as idle, regular use, and high consumption scenarios. While KMeans does not offer a direct predictive utility like regression models, it adds value by uncovering latent structure in user behavior and consumption patterns, which could be useful for segmenting households or optimizing energy delivery strategies.

Overall, linear regression emerged as the most effective model for precise prediction in this task, thanks to its simplicity, speed, and high accuracy on the given data. Decision trees offered comparable results and serve as a robust backup model, especially in real-world scenarios where linearity cannot be assumed. KMeans clustering provided important exploratory insights, complementing the regression models by identifying patterns that are not immediately obvious through prediction alone. Together, these models demonstrate a well-rounded approach to both predictive and exploratory analysis of time-series energy data.

