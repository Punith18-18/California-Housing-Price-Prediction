## 🏠 California Housing Price Prediction

This project focuses on predicting California housing prices using various regression models. It includes exploratory data analysis (EDA), model comparison, and performance visualization.

## 📊 Project Highlights

- Built an end-to-end ML pipeline using Python, Pandas, and Scikit-learn.
- Performed detailed EDA with histograms, scatter plots, and correlation matrices.
- Trained and compared multiple models including:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluated models using:
  - Mean Squared Error (MSE)
  - R² Score
- Achieved an R² Score of **0.81** with the Random Forest model.
- Visualized:
  - Feature importance
  - Predicted vs Actual values
  - Median income vs Median house value
  - Latitude vs Median house value
  - Residual plots (optional)
- Scaled features using `StandardScaler` for better model performance.

## 📁 Dataset

- Used California Housing dataset from `sklearn.datasets` (also available via Google Colab).
- Features include income, rooms, bedrooms, population, location coordinates, etc.

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

## 🚀 Future Improvements

- Deploy model using Streamlit or Flask
- Integrate with Power BI or Tableau for real-time dashboards
- Save and load models using `joblib` for deployment

## 📸 Sample Visuals

| 📍 Visual | Description |
|----------|-------------|
| ✅ Histograms | Distribution of housing data features |
| ✅ Correlation Matrix | Identify strong/weak correlations |
| ✅ Scatter Plots | Visual relation between features and house value |
| ✅ Model Comparison | R² scores and MSE across models |
| ✅ Feature Importance | Most influential features in prediction |
