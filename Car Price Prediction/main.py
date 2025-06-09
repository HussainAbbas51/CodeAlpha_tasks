# Complete Car Price Prediction Project Code

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load dataset
df = pd.read_csv("car data.csv")

# Step 3: Data Preprocessing
df['Current_Year'] = 2025
df['Car_Age'] = df['Current_Year'] - df['Year']
df.drop(['Year', 'Car_Name', 'Current_Year'], axis=1, inplace=True)

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Step 4: Feature and target split
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

evaluation_results = {
    "Mean Absolute Error": mae,
    "Mean Squared Error": mse,
    "Root Mean Squared Error": rmse,
    "R^2 Score": r2
}

# Step 9: Plot actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

evaluation_results

