import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and preprocess the data
def load_data(filepath):
    # Load only necessary rows to avoid memory overload
    df = pd.read_csv(filepath, sep=';', low_memory=False, na_values='?', parse_dates=[[0, 1]],
                     infer_datetime_format=True, index_col='Date_Time')

    df = df.dropna()
    
    # Convert columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only a few features for simplicity
    df = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
    df = df.dropna()

    return df

# Step 2: Split into features and target
def prepare_data(df):
    X = df.drop('Global_active_power', axis=1)
    y = df['Global_active_power']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train and evaluate regression models
def evaluate_model(name, model, X_test, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - RMSE: {rmse:.3f}, R² Score: {r2:.3f}")

def regression_models(X_train, X_test, y_train, y_test):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    evaluate_model("Linear Regression", lr, X_test, y_test, y_pred_lr)

    # Decision Tree Regressor
    dt = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    evaluate_model("Decision Tree", dt, X_test, y_test, y_pred_dt)

# Step 4: Apply KMeans clustering
def clustering(df):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    df['Cluster'] = labels

    # Plot clustering result
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='Voltage', y='Global_intensity', hue='Cluster', data=df, palette='Set2')
    plt.title('KMeans Clustering of Electricity Usage')
    plt.show()

# Run everything
def main():
    filepath = 'household_power_consumption.txt'  # Update path if needed
    df = load_data(filepath)
    print("Data loaded and cleaned.")

    X_train, X_test, y_train, y_test = prepare_data(df)
    print("Data split into training and test sets.")

    print("\nRegression Model Results:")
    regression_models(X_train, X_test, y_train, y_test)

    print("\nRunning KMeans clustering:")
    clustering(df)

if __name__ == '__main__':
    main()
