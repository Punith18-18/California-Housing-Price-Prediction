# California-Housing-Price-Prediction
Overview
This project aims to predict the median house value in California using the 1990 U.S. Census housing data. The dataset contains information such as location (latitude and longitude), housing age, number of rooms and bedrooms, population, households, and median income. Our objective is to build a robust, production-ready machine learning model to accurately forecast housing prices.

Dataset
The California Housing dataset includes the following key features:

longitude: Longitudinal coordinate of the housing block.
latitude: Latitudinal coordinate of the housing block.
housingMedianAge: Median age of the houses.
totalRooms: Total number of rooms.
totalBedrooms: Total number of bedrooms.
population: Population of the housing block.
households: Number of households.
medianIncome: Median income of households (in tens of thousands).
medianHouseValue: The target variable representing the median house value (USD).
Project Structure
bash
Copy
Edit
.
├── README.md              # Project overview and documentation
├── california_housing_train.csv  # Training data file
├── california_housing_test.csv   # Test data file
├── model.py               # Script containing modularized functions for loading data, training, and evaluation
└── notebooks/             # Jupyter or Colab notebooks for exploration and visualization
Setup and Execution
Running the Project on Google Colab
Upload Data Files:
In your Google Colab environment, upload the california_housing_train.csv and california_housing_test.csv files.

python
Copy
Edit
from google.colab import files
uploaded = files.upload()  # Upload the CSV files
Load and Explore Data:
Use pandas to load the data and perform initial exploration.

python
Copy
Edit
import pandas as pd

# Load the datasets
train_data = pd.read_csv('california_housing_train.csv')
test_data = pd.read_csv('california_housing_test.csv')

# Preview the data
print(train_data.head())
print(train_data.describe())
Visualize Data:
Generate histograms, correlation matrices, and scatter plots to understand data distributions and relationships.

python
Copy
Edit
import matplotlib.pyplot as plt
import seaborn as sns

# Histograms
train_data.hist(bins=50, figsize=(20,15))
plt.suptitle("Histograms of California Housing Features", fontsize=20)
plt.show()

# Correlation Matrix
plt.figure(figsize=(12,10))
sns.heatmap(train_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix", fontsize=18)
plt.show()

# Scatter Plot: Median Income vs. Median House Value
plt.figure(figsize=(8,6))
plt.scatter(train_data['medianIncome'], train_data['medianHouseValue'], alpha=0.3)
plt.xlabel("Median Income (in tens of thousands)")
plt.ylabel("Median House Value (USD)")
plt.title("Median Income vs. Median House Value")
plt.show()
Code Improvements and Best Practices
1. Modular Code Structure
Break your script into functions to improve maintainability:

python
Copy
Edit
def load_data(train_path, test_path):
    import pandas as pd
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def visualize_data(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df.hist(bins=50, figsize=(20,15))
    plt.suptitle("Histograms of California Housing Features", fontsize=20)
    plt.show()
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix", fontsize=18)
    plt.show()
2. Scikit-Learn Pipelines
Utilize pipelines for streamlined preprocessing and model training:

python
Copy
Edit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
3. Hyperparameter Tuning with GridSearchCV
Optimize model parameters for better performance:

python
Copy
Edit
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
4. Enhanced Evaluation Metrics and Visualizations
Evaluate your model using multiple metrics and plots:

python
Copy
Edit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    # Predicted vs. Actual
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Predicted vs. Actual")
    plt.show()

    # Residual Plot
    plt.figure(figsize=(8,6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()
5. Error Handling and Logging
Implement logging and error handling to aid in debugging:

python
Copy
Edit
import logging

logging.basicConfig(level=logging.INFO)

try:
    train_data, test_data = load_data('california_housing_train.csv', 'california_housing_test.csv')
    logging.info("Data loaded successfully!")
except Exception as e:
    logging.error("Error loading data: " + str(e))
6. Saving the Trained Model
Persist your trained model for later use:

python
Copy
Edit
import joblib
joblib.dump(grid_search.best_estimator_, 'california_housing_model.pkl')
7. Additional Feature Engineering
Enhance your dataset with new features such as "rooms per household":

python
Copy
Edit
train_data['roomsPerHousehold'] = train_data['totalRooms'] / train_data['households']
test_data['roomsPerHousehold'] = test_data['totalRooms'] / test_data['households']

features = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms', 
            'totalBedrooms', 'population', 'households', 'medianIncome', 'roomsPerHousehold']

X_train = train_data[features]
y_train = train_data['medianHouseValue']
X_test = test_data[features]
y_test = test_data['medianHouseValue']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Next Steps and Enhancements
Model Experimentation: Try other models like Linear Regression, XGBoost, or ensemble methods.
Advanced Hyperparameter Tuning: Experiment with techniques like RandomizedSearchCV or Bayesian optimization.
Cross-Validation: Use cross-validation to validate model performance and ensure robustness.
Deployment: Consider deploying your model as a web application using frameworks like Flask or Streamlit.
Documentation & Reporting: Extend the documentation and create reports/presentations summarizing your analysis and findings.
Conclusion
This project demonstrates a full machine learning workflow—from data loading and exploration to model training, hyperparameter tuning, evaluation, and deployment preparation. By following best practices such as modular code design, the use of pipelines, rigorous evaluation, and logging, this project serves as a solid foundation for building production-ready machine learning applications.

Enjoy building and enhancing your model!

