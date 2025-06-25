#WEEK 2
1. FashionMNIST

For this project, I implemented a convolutional neural network (CNN) using PyTorch to classify images from the FashionMNIST dataset — a collection of 28×28 grayscale images across 10 clothing categories. PyTorch was chosen for its dynamic computation graph and flexibility, making it ideal for developing and debugging deep learning models at a conceptual level, which aligns well with my academic setting and goals.
The model architecture consists of two convolutional layers with Batch Normalization, ReLU activations, and MaxPooling, followed by a fully connected classification head with Dropout regularization to reduce overfitting. I used the Adam optimizer for efficient gradient updates, and a StepLR scheduler to reduce the learning rate halfway through training for better convergence. Data was normalized to mean 0.5 and standard deviation 0.5 to stabilize learning.
The model was trained for 10 epochs on a batch size of 64, and achieved a test accuracy of over 92%, which is a strong result for FashionMNIST. The loss curve plotted across epochs demonstrated steady and consistent convergence, indicating effective learning without signs of overfitting or vanishing gradients.
To support qualitative understanding, I visualized sample predictions alongside their ground truth labels. Most predictions were accurate, and even the incorrect ones (e.g., a coat being classified as a shirt) were justifiable given visual similarity — reflecting the model's real-world reasoning limitations. I also included a confusion matrix and a detailed classification report to break down the model's performance across all categories, with especially strong precision and recall for classes like trousers, bags, and sneakers.
Finally, I saved both the model and visual artifacts (loss curve, confusion matrix, and predictions), ensuring reproducibility and professional presentation. 

<img width="451" alt="fashionoutput" src="https://github.com/user-attachments/assets/380950ec-e7a3-4006-a3cc-425de6a0e821" />
<img width="911" alt="fashionreport" src="https://github.com/user-attachments/assets/9f895361-35a4-470f-8ce6-c3e1332188e6" />
confusion matrix image: ![confusion_matrix](https://github.com/user-attachments/assets/936ea271-406d-4955-88d1-d96177dbf63b)

loss curve image: ![loss_curve](https://github.com/user-attachments/assets/a7e024cc-b7bd-4678-a8ab-02aae188bd40)

sample predictions image: ![sample_predictions](https://github.com/user-attachments/assets/310082b7-abf2-4fc4-bf48-3d153103be05)




#WEEK 1
TASK 1 NOTES

Source: UCI Heart Disease Dataset (Cleveland) https://archive.ics.uci.edu/dataset/45/heart+disease

This machine learning project aimed to build and compare multiple classification models to predict the presence of heart disease using the Cleveland dataset. The target variable, "num", was converted into a binary outcome: 0 for absence of disease and 1 for presence. After handling missing values and scaling the features, four models were implemented and evaluated: Logistic Regression, Random Forest, Support Vector Classifier (SVC), and XGBoost.

Among these models, the Random Forest Classifier emerged as the best performer. It achieved the highest accuracy of 88.5% and the highest ROC AUC score of 0.959. This indicates that it not only made the most correct predictions but also had excellent ability to distinguish between the positive and negative classes across all threshold values. Random Forest's strength lies in its ensemble nature, where multiple decision trees contribute to a more robust and generalized prediction.

The Support Vector Classifier also performed strongly, matching the Random Forest in accuracy with 88.5% but slightly lower in ROC AUC at 0.944. This suggests that while it was equally accurate overall, its probability estimates were not as well calibrated for classification thresholds. Nevertheless, its high precision and recall scores for both classes indicate it is a strong model, especially for problems where margin-based separation is effective.

Logistic Regression and XGBoost both achieved slightly lower accuracy scores of 85.2%. However, Logistic Regression had an impressive ROC AUC of 0.958, suggesting that it performed well in ranking predictions even if some classifications were incorrect. XGBoost, despite its reputation for high performance in many scenarios, lagged slightly behind here with a ROC AUC of 0.927. This might be due to the relatively small size of the dataset or the fact that the optimal hyperparameters for XGBoost were more difficult to pinpoint given the tuning grid.

In summary, Random Forest was the most effective model in this context due to its combination of high accuracy and strong ROC AUC score. It handled the feature interactions and potential non-linearities in the data better than the other models. However, all models performed reasonably well, and the consistent results across different approaches suggest that the dataset is well-suited for binary classification with the applied preprocessing steps. 



TASK 2 NOTES

I worked with the “Household Power Consumption” dataset, which contains time-series data on household electricity usage. The original dataset had several columns and some missing values. To ensure efficient processing and focus on meaningful patterns, I first cleaned the data by dropping missing values and selecting only four core features: Global Active Power, Global Reactive Power, Voltage, and Global Intensity. These were converted to numeric types, and only valid rows were retained. After this, I randomly sampled 50,000 rows from the dataset to reduce computational load while maintaining statistical integrity.

Next, I separated the features from the target variable, which was Global Active Power. The features were scaled using standardization to ensure all values contributed equally to model training. The data was then split into training and test sets using an 80-20 split.

I implemented and evaluated three different regression models: Linear Regression, Decision Tree Regressor, and Random Forest Regressor. Linear Regression achieved an RMSE of 0.042 and an R² score of 0.998, indicating extremely high accuracy and minimal prediction error. The Decision Tree model produced a slightly higher RMSE of 0.055 and an R² of 0.997, still strong but marginally less accurate. Random Forest, which is an ensemble model based on multiple decision trees, performed nearly as well as Linear Regression, with an RMSE of 0.047 and R² of 0.998.

From these results, the Linear Regression and Random Forest models performed the best. Linear Regression was the most efficient and surprisingly accurate, suggesting that the relationship between the features and the target is largely linear. Random Forest also performed very well, benefiting from ensemble averaging and reduced overfitting compared to a single tree. The Decision Tree, while still good, was the weakest of the three, possibly due to limited depth and sensitivity to small variations in the data.

Lastly, I applied KMeans clustering on the same dataset using three clusters. The dataset was standardized before clustering. The resulting clusters were visualized using a scatterplot of Voltage vs Global Intensity, colored by cluster label. This allowed us to explore patterns in the electricity usage without a supervised target.

Overall, the workflow involved structured preprocessing, applying multiple ML models, performance evaluation using RMSE and R² metrics, and an unsupervised clustering step. Among the regression models, Linear Regression was the top performer in this case, suggesting a strong linear relationship in the data. The analysis and comparison of different models helped understand how model complexity affects performance on a real-world dataset.


