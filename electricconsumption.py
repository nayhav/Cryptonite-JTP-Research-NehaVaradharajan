import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Step 1: Load and clean data
def load(path):
    df = pd.read_csv(path, sep=';', low_memory=False, na_values='?', parse_dates=[[0, 1]],
                     infer_datetime_format=True, index_col='Date_Time')
    df = df.dropna()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
    df = df.dropna()
    return df

# Step 2: Prepare data for training
def prepare(df):
    x = df.drop('Global_active_power', axis=1)
    y = df['Global_active_power']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Step 3: Evaluate and run models
def score(name, model, xtest, ytest, ypred):
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    r2 = r2_score(ytest, ypred)
    print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")
    return rmse, r2

def runmodels(xtrain, xtest, ytrain, ytest):
    results = {}

    model1 = LinearRegression()
    model1.fit(xtrain, ytrain)
    pred1 = model1.predict(xtest)
    print("Linear done")
    results["Linear"] = score("Linear", model1, xtest, ytest, pred1)

    model2 = DecisionTreeRegressor(max_depth=6, random_state=42)
    model2.fit(xtrain, ytrain)
    pred2 = model2.predict(xtest)
    print("Tree done")
    results["Tree"] = score("Tree", model2, xtest, ytest, pred2)

    model3 = RandomForestRegressor(n_estimators=10, random_state=42)
    model3.fit(xtrain, ytrain)
    pred3 = model3.predict(xtest)
    print("Forest done")
    results["Forest"] = score("Forest", model3, xtest, ytest, pred3)

    return results

# Step 4: Clustering
def cluster(df):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(data_scaled)
    df['Cluster'] = labels

    print("Clustering done (plot disabled for now)")
    # Optional: enable below if needed
    # plt.figure(figsize=(8, 4))
    # sns.scatterplot(x='Voltage', y='Global_intensity', hue='Cluster', data=df, palette='Set2')
    # plt.title('KMeans Clustering')
    # plt.show()

# Main function
def main():
    path = 'household_power_consumption.txt'  # update path if needed
    df = load(path)
    df = df.sample(n=50000, random_state=42)  # reduce data for speed
    print("Data loaded and sampled")

    xtrain, xtest, ytrain, ytest = prepare(df)
    print("Data split")

    print("\nModel results:")
    results = runmodels(xtrain, xtest, ytrain, ytest)

    print("\nClustering:")
    cluster(df)

    print("\nFinal scores:")
    for name in results:
        rmse, r2 = results[name]
        print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")

if __name__ == '__main__':
    main()
