# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules for ML models and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

#Loading and clean the dataset
def load(path):
    # Loading CSV file, combine Date and Time into a single datetime column
    df = pd.read_csv(path, sep=';', low_memory=False, na_values='?', parse_dates=[[0, 1]],
                     infer_datetime_format=True, index_col='Date_Time')

    # Dropping missing values to avoid errors during training
    df = df.dropna()

    # Converting all values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keeping only key features for this project
    df = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
    df = df.dropna()
    return df

#Preparing the data for training and testing
def prepare(df):
    # Separate features (input variables) and target (what we want to predict)
    x = df.drop('Global_active_power', axis=1)
    y = df['Global_active_power']

    # Standardize the features to have mean=0 and std=1
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Split into 80% training and 20% test data
    return train_test_split(x_scaled, y, test_size=0.2, random_state=42)

#Evaluate a model's predictions using RMSE and R2
def score(name, model, xtest, ytest, ypred):
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    r2 = r2_score(ytest, ypred)
    print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")
    return rmse, r2

# Train and test three regression models
def runmodels(xtrain, xtest, ytrain, ytest):
    results = {}

    # Linear Regression
    model1 = LinearRegression()
    model1.fit(xtrain, ytrain)
    pred1 = model1.predict(xtest)
    print("Linear done")
    results["Linear"] = score("Linear", model1, xtest, ytest, pred1)

    # Decision Tree Regressor
    model2 = DecisionTreeRegressor(max_depth=6, random_state=42)
    model2.fit(xtrain, ytrain)
    pred2 = model2.predict(xtest)
    print("Tree done")
    results["Tree"] = score("Tree", model2, xtest, ytest, pred2)

    # Random Forest Regressor
    model3 = RandomForestRegressor(n_estimators=10, random_state=42)
    model3.fit(xtrain, ytrain)
    pred3 = model3.predict(xtest)
    print("Forest done")
    results["Forest"] = score("Forest", model3, xtest, ytest, pred3)

    return results

# Apply unsupervised learning (KMeans clustering)
def cluster(df):
    # Standardize features before clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Apply KMeans clustering with 3 clusters
    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(data_scaled)
    df['Cluster'] = labels  

    print("Clustering done")

    # Plot the clusters
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x='Voltage', y='Global_intensity', hue='Cluster', data=df, palette='Set2')
    plt.title('KMeans Clustering')
    plt.xlabel('Voltage')
    plt.ylabel('Global Intensity')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

# Main workflow
def main():
    path = 'household_power_consumption.txt'
    df = load(path)
    
    # Using a subset of the data to reduce computation time
    df = df.sample(n=50000, random_state=42)
    print("Data loaded and sampled")

    # Split data into training and testing sets
    xtrain, xtest, ytrain, ytest = prepare(df)
    print("Data split")

    # Train and evaluate regression models
    print("\nModel results:")
    results = runmodels(xtrain, xtest, ytrain, ytest)

    # Apply and visualize clustering
    print("\nClustering:")
    cluster(df)

    # Display final summary of model performance
    print("\nFinal scores:")
    for name in results:
        rmse, r2 = results[name]
        print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")

# Execute the program
if __name__ == '__main__':
    main()
